import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, embedding_size: int=150):

        super().__init__()

        exponential = (torch.arange(0, embedding_size, 2) * -2) / embedding_size
        freq_denom = torch.pow(10000, exponential)
        self.register_buffer('freq_denom', freq_denom)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        freq = x[..., None] * self.freq_denom[None, ...]
        return torch.concat([torch.sin(freq), torch.cos(freq)], dim=-1)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()

        # assets
        self.volcab_size = n_tokens

        # input handling
        self.embeddings = torch.nn.Embedding(self.volcab_size, d_latent)
        self.positional_embeddings = PositionalEmbedding(d_latent)
        
        self.bos_embedding = torch.nn.Parameter(torch.empty(1, 1, d_latent))
        
        # Fills the tensor in-place using a uniform distribution
        torch.nn.init.xavier_uniform_(self.bos_embedding)

        #  transformer
        decoder_layer = torch.nn.TransformerEncoderLayer(d_latent, nhead=8, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(decoder_layer, num_layers=6)

        # output
        self.output_mlp = torch.nn.Linear(d_latent, self.volcab_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        
        # flatten per batch
        x = x.flatten(start_dim=1)

        # embed the tokens and add positional embeddings
        x = self.embeddings(x) + self.positional_embeddings(x)

        # expand bos embedding
        batch_size = x.size(dim=0)
        bos_vectors = self.bos_embedding.expand(batch_size, -1, -1)

        # shift the embeddings by 1, remove the last embedding
        x = torch.cat((bos_vectors, x[:, :-1, :]), dim=1)
        
        # generate a look ahead mask for the inputs
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
                                                x.size(1)
                                            ).to(
                                                x.device
                                            )
        # pass it through transformer    
        x = self.transformer(x, mask=mask, is_causal=True)

        # get logits and reshape to image format
        logits = self.output_mlp(x).view(-1, 20, 30, self.volcab_size)

        return logits, {}

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa

        batches = torch.tensor([]).to(device)
    
        for _ in range(B):

            grid = torch.full((h, w), 0).to(device)

            for h_idx in range(h):

                for w_idx in range(w):

                    grid_logits = self.forward(grid)[0]
                    predictions = torch.squeeze(torch.argmax(grid_logits, dim=-1), dim=0)
                    grid[h_idx, w_idx] = predictions[h_idx, w_idx]
            
            if batches.size(dim=0) == 0:
        
                batches = grid[None, :]
                continue
                
            batches = torch.vstack((batches, grid[None, :]))
                
        return batches
