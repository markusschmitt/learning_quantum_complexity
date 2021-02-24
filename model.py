#
# Definition of autoencoder networks
#
# Author: Markus Schmitt
# Date: Feb 2021
#

import torch
from torch import nn

class BottleneckNet(nn.Module):
    """This class defines the autoencoder network

    Initializer arguments:

        * ``num_latent``: Number of latent variables (integer).
        * ``encoder_dim``: Sizes of the encoder layers (list of integers)
        * ``decoder_dim``: Sizes of the decoder layers (list of integers)
    """

    def __init__(self, num_latent, input_dim, encoder_dim=[100,100], decoder_dim=[100,100]):
        """Initializes ``BottleneckNet``.

        Arguments:

            * ``num_latent``: Number of latent variables (integer).
            * ``input_dim``: Dimension of input (integer).
            * ``encoder_dim``: Sizes of the encoder layers (list of integers)
            * ``decoder_dim``: Sizes of the decoder layers (list of integers)
        """

        super().__init__()

        # List of encoder layers
        self.enc = [nn.Linear(input_dim, encoder_dim[0])]
        for k in range(len(encoder_dim)-2):
            self.enc.append(nn.Linear(encoder_dim[k],encoder_dim[k+1]))
        self.enc.append(nn.Linear(encoder_dim[-1], num_latent))

        self.enc = nn.ModuleList(self.enc)
        
        # List of decoder layers
        self.dec = [nn.Linear(num_latent, decoder_dim[0])]
        for k in range(len(decoder_dim)-2):
            self.dec.append(nn.Linear(decoder_dim[k],decoder_dim[k+1]))
        self.dec.append(nn.Linear(decoder_dim[-1], input_dim))
        
        self.dec = nn.ModuleList(self.dec)

    def forward(self, x):
        """Network evaluation.

        Arguments:

            * ``x``: Input data (Pytorch tensor of floats)

        Returns: Network output (same dimensions as input), latent values
        """

        # Encoder
        for layer in self.enc:
            x=layer(x)
            x=torch.tanh(x)

        # Store latent values
        latent=x.clone()

        # Decoder
        for layer in self.dec:
            x=layer(x)
            x=torch.tanh(x)

        return x, latent
    
    def compute_latent(self, x):
        """Compute latent values.

        Evaluate only the encoder part of the network

        Arguments:

            * ``x``: Input data (Pytorch tensor of floats)

        Returns: latent values
        """

        # Encoder
        for layer in self.enc:
            x=layer(x)
            x=torch.tanh(x)
        return x

class NoLatentNet(nn.Module):
    """This class mimics the ''autoencoder'' network without latent variables.

    Initializer arguments:

        * ``input_dim``: Dimension of input (integer).
    """
    def __init__(self, input_dim):
        """Initialize ``NoLatentNet``

        Arguments:

            * ``input_dim``: Dimension of input (integer).
        """

        super().__init__()

        # Parameters W correspond to the constant output.
        self.W = torch.nn.Parameter(torch.randn(input_dim))
        self.W.requires_grad = True

    def forward(self, x):
        """Network evaluation.

        The network output is independent of the input.

        Arguments:

            * ``x``: Input data (Pytorch tensor of floats)
        
        Returns: Network output (same dimensions as input), latent values (dummy)
        """

        return torch.stack([self.W for i in x]), torch.tensor([0.])
