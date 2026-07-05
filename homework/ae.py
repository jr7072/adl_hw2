from multiprocessing import Value
from unittest.mock import patch
import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """

class EncoderBlock(torch.nn.Module):

    def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    stride: int=1,
                    padding: int=0,
                    non_linearity: torch.nn.Module = torch.nn.Identity # ty: ignore
                ):

        super().__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            non_linearity()
        ]

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class DecoderBlock(torch.nn.Module):

    def __init__(self,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    stride: int=1,
                    padding: int=0,
                    output_padding: tuple[int, int]=(0, 0),
                    non_linearity: torch.nn.Module=torch.nn.Identity # ty: ignore
                 ):

        super().__init__()

        layers = [
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            ),
            non_linearity()
        ]

        self.block = torch.nn.Sequential(
            *layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchEncoder(torch.nn.Module):
    """
    (Optionally) Use this class to implement an encoder.
                    It can make later parts of the homework easier (reusable components).
    """

    def __init__(self, latent_dim: int, latent_shape: tuple[int, int]):

        super().__init__()

        layers = [
            EncoderBlock(in_channels=3,
                            out_channels=latent_dim // 4,
                            kernel_size=2,
                            stride=2,
                            non_linearity=torch.nn.GELU # ty: ignore
                        ),
            EncoderBlock(in_channels=latent_dim // 4,
                            out_channels=latent_dim // 2,
                            kernel_size=3,
                            stride=2,
                            non_linearity=torch.nn.GELU # ty: ignore
                        ),
            EncoderBlock(in_channels=latent_dim // 2,
                            out_channels=latent_dim,
                            kernel_size=3,
                            non_linearity=torch.nn.GELU # ty: ignore
                        ),
            torch.nn.AdaptiveAvgPool2d(output_size=latent_shape)
            
        ]

        self.encoder = torch.nn.Sequential(
            *layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # transpose the tensor
        transposed_vector = hwc_to_chw(x)
        return self.encoder(transposed_vector)
        
class PatchDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, latent_out_shape: tuple[int, int]):

        super().__init__()

        layers = [
            torch.nn.AdaptiveAvgPool2d(output_size=latent_out_shape),
            DecoderBlock(
                in_channels=latent_dim,
                out_channels=latent_dim // 2,
                kernel_size=3,
                non_linearity=torch.nn.GELU # ty: ignore
            ),
            DecoderBlock(
                in_channels=latent_dim // 2,
                out_channels=latent_dim // 4,
                kernel_size=3,
                stride=2,
                output_padding=(1, 0),
                non_linearity=torch.nn.GELU # ty: ignore
            ),
            DecoderBlock(
                in_channels=latent_dim // 4,
                out_channels=3, # back to original channels
                kernel_size=2,
                stride=2
            )
        ]

        self.decoder = torch.nn.Sequential(
            *layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = self.decoder(x)
        return chw_to_hwc(y)


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    
    def __init__(self, patch_size: int = 5, latent_dim: int = 128):
        super().__init__()

        if (100 % patch_size != 0) or (150 % patch_size != 0):
            raise ValueError("invalid patch size")
        
        latent_shape = (100 // patch_size, 150 // patch_size)
        latent_out_shape = (22, 35)

        self.encoder = PatchEncoder(latent_dim=latent_dim,
                                        latent_shape=latent_shape)
        self.decoder = PatchDecoder(latent_dim=latent_dim,
                                        latent_out_shape=latent_out_shape)

        self.model = torch.nn.Sequential(
            self.encoder,
            self.decoder
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        y = self.model(x)
        return y, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
