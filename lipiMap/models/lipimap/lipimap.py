from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

from .modules import MaskedLinearDecoder, ExtEncoder, ZeroOneClipper
from ..trvae.losses import mse, nb
from .losses import hsic
from ..trvae._utils import one_hot_encoder
from ..base._base import CVAELatentsModelMixin

# Adapted from
# Title: Biologically informed deep learning to query gene programs in single-cell atlases
# Authors: Mohammad Lotfollahi, Sergei Rybakov, Karin Hrovatin, Soroor Hediyeh-zadeh, Carlos Talavera-López, Alexander V. Misharin & Fabian J. Theis 
# Code: https://github.com/theislab/scarches/tree/master/scarches/models/expimap/expimap.py

class lipiMap(nn.Module, CVAELatentsModelMixin):
    """lipiMap model class. This class contains the implementation of Conditional Variational Auto-encoder.
       Parameters
       ----------
       input_dim: Integer
            Number of input features (i.e. gene in case of scRNA-seq).
       conditions: List
            List of Condition names that the used data will contain to get the right encoding when used after reloading.
       hidden_layer_sizes: List
            A list of hidden layer sizes for encoder network. Decoder network will be the reversed order.
       latent_dim: Integerfrom 
            Bottleneck layer (z) size.
       dr_rate: Float
            Dropput rate applied to all layers, if `dr_rate`=0 no dropout will be applied.
       recon_loss: String
            Definition of Reconstruction-Loss-Method, 'mse' or 'nb'.
       use_l_encoder: Boolean
            If True and `decoder_last_layer`='softmax', libary size encoder is used.
       use_bn: Boolean
            If `True` batch normalization will be applied to layers.
       use_ln: Boolean
            If `True` layer normalization will be applied to layers.
       mask: Tensor or None
            if not None, Tensor of 0s and 1s from utils.add_annotations to create VAE with a masked linear decoder.
       decoder_last_layer: String or None
            The last layer of the decoder. Must be 'softmax' (default for 'nb' loss), identity(default for 'mse' loss),
            'softplus', 'exp' or 'relu'.
       TODO: add more parameters to the docstring (e.g.clipping_decoder_weights)  
    """

    def __init__(self,
                 input_dim: int,                            # Input dimension (number of lipids)
                 latent_dim: int,                           # Latent dimension (number of GPs)
                 mask: torch.Tensor,                        # Lipid-LP matrix for the masked linear decoder (n_GPs x n_genes)
                 condition_key: str,                        # Key for the condition column in the AnnData object or 'arbitrary' key for the condition column TODO: documentation
                 conditions: list,                          # List of all the possible values of the conditioning variable
                 hidden_layer_sizes: list = [256, 256],
                 dr_rate: float = 0.05,                     # Dropout rate
                 recon_loss: str = 'mse',                   # Reconstruction loss: 'mse' or 'nb'
                 use_l_encoder: bool = False,               # Use 'library size encoder'
                 use_bn: bool = False,                      # Batch Normalization
                 use_ln: bool = True,                       # Layer Normalization
                 decoder_last_layer: Optional[str] = None,  # 'softmax' for nb-loss, 'identity' for mse-loss
                 soft_mask: bool = False,                   # Soft-membership Regularization
                 n_ext: int = 0,                            # Number of Unconstrained extension nodes (Architecture Surgery for Query Mapping --> ASQM)
                 n_ext_m: int = 0,                          # Number of Constrained extension nodes (ASQM, m stands for masked)
                 use_hsic: bool = False,                    # HSIC (Hilbert-Schmidt Independence Criterion) Regularization (for unconstrained extension nodes)
                 hsic_one_vs_all: bool = False,
                 ext_mask: Optional[torch.Tensor] = None,   # Gene-GP matrix for the masked linear decoder for extension nodes
                 soft_ext_mask: bool = False,               # Soft-membership Regularization for extension nodes
                 clipping_decoder_weights: bool = False,    # Clip decoder weights to [0, 1]
                #  representation_score_correction: bool = False, # Representation Score Correction
                #  rho: float = 1.0,                         # Rho parameter for the representation score correction
                 ):
        
        super().__init__()
        assert isinstance(hidden_layer_sizes, list)
        assert isinstance(latent_dim, int)
        assert isinstance(conditions, list)
        assert isinstance(condition_key, str) or condition_key is None
        assert recon_loss in ["mse"], "'recon_loss' must be 'mse'"

        if condition_key == 'arbitrary':
            assert len(conditions) > 0, "If condition_key is 'arbitrary', conditions must be provided."

        print("\nINITIALIZING NEW NETWORK..............")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_conditions = len(conditions)
        self.condition_key = condition_key
        self.conditions = conditions
        self.condition_encoder = {k: v for k, v in zip(conditions, range(len(conditions)))} if condition_key != 'arbitrary' else None
        self.recon_loss = recon_loss
        self.freeze = False
        self.use_bn = use_bn
        self.use_ln = use_ln

        self.n_ext_encoder = n_ext + n_ext_m
        self.n_ext_decoder = n_ext
        self.n_ext_m_decoder = n_ext_m

        self.use_hsic = use_hsic and self.n_ext_decoder > 0
        self.hsic_one_vs_all = hsic_one_vs_all

        self.soft_mask = soft_mask and mask is not None
        self.soft_ext_mask = soft_ext_mask and ext_mask is not None

        self.clipping_decoder_weights = clipping_decoder_weights
        # self.representation_score_correction = representation_score_correction
        # self.rho = torch.Tensor(rho) if representation_score_correction else None

        if decoder_last_layer is None:
            self.decoder_last_layer = 'identity'
        else:
            self.decoder_last_layer = decoder_last_layer

        self.use_l_encoder = use_l_encoder

        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Encoder
        encoder_layer_sizes = self.hidden_layer_sizes.copy()
        encoder_layer_sizes.insert(0, self.input_dim)

        self.encoder = ExtEncoder(encoder_layer_sizes,
                                  self.latent_dim,
                                  self.use_bn,
                                  self.use_ln,
                                  self.use_dr,
                                  self.dr_rate,
                                  self.n_conditions,
                                  self.condition_key, ###########################################
                                  self.n_ext_encoder)

        if self.soft_mask:
            self.n_inact_genes = (1-mask).sum().item()
            soft_shape = mask.shape
            if soft_shape[0] != latent_dim or soft_shape[1] != input_dim:
                raise ValueError('Incorrect shape of the soft mask.')
            self.mask = mask.t()
            mask = None
        else:
            self.mask = mask # ??? self.mask=None in expiMap... anyway, 'mask' is passed to the decoder, and not 'self.mask'

        if self.soft_ext_mask:
            self.n_inact_ext_genes = (1-ext_mask).sum().item()
            ext_shape = ext_mask.shape
            if ext_shape[0] != self.n_ext_m_decoder:
                raise ValueError('Dim 0 of ext_mask should be the same as n_ext_m_decoder.')
            if ext_shape[1] != self.input_dim:
                raise ValueError('Dim 1 of ext_mask should be the same as input_dim.')
            self.ext_mask = ext_mask.t()
            ext_mask = None
        else:
            self.ext_mask = ext_mask # self.ext_mask=None in expiMap... anyway, 'ext_mask' is passed to the decoder, and not 'self.ext_mask'

        self.decoder = MaskedLinearDecoder(self.latent_dim,
                                           self.input_dim,
                                           self.n_conditions,
                                           self.condition_key, ###########################################
                                           mask,
                                           ext_mask,
                                           self.recon_loss,
                                           self.decoder_last_layer,
                                           self.n_ext_decoder,
                                           self.n_ext_m_decoder)
        if self.clipping_decoder_weights:
            self.clipper = ZeroOneClipper()
            self.decoder.apply(self.clipper)

        if self.use_l_encoder:
            self.l_encoder = ExtEncoder([self.input_dim, 128],
                                        1,
                                        self.use_bn,
                                        self.use_ln,
                                        self.use_dr,
                                        self.dr_rate,
                                        self.n_conditions)

    def forward(self, a, b, k, x=None, batch=None, sizefactor=None, labeled=None):
        """
        Forward pass for lipiMap, which includes encoding, sampling, decoding, 
        and calculating the relevant losses (reconstruction, KL divergence, and HSIC).

        Parameters
        ----------
        a : float
            Lower threshold for the weighting mask.
        b : float
            Upper threshold for the weighting mask.
        k : float
            Weighting factor used to scale the loss for values outside the range [a, b].
        x : torch.Tensor, optional
            Input data tensor of shape `(batch_size, n_features)`. Default is `None`.
        batch : torch.Tensor, optional
            Batch assignment tensor of shape `(batch_size,)`, used for handling 
            conditioning in the encoder and decoder. Default is `None`.

        Returns
        -------
        tuple
            A tuple containing the following losses:
            - `recon_loss` (torch.Tensor): The mean squared error (MSE) reconstruction loss.
            - `weighted_recon_loss` (torch.Tensor): The weighted reconstruction loss, 
            where weighting scales the reconstruction error for values outside [a, b].
            - `kl_div` (torch.Tensor): The KL divergence between the posterior and prior distributions.
            - `hsic_loss` (torch.Tensor): The HSIC (Hilbert-Schmidt Independence Criterion) loss, 
            which quantifies dependence between annotated and external latent variables.
        """
        assert k >= 0
        # x_log = torch.log(1 + x)
        # x_log = x

        z1_mean, z1_log_var = self.encoder(x, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        
        outputs = self.decoder(z1, batch)
        
        # Reconstruction loss
        recon_x, y1 = outputs
        recon_loss = mse(recon_x, x).sum(dim=-1).mean()
        weighting_mask = ((x < a) | (x > b))
        weighted_recon_loss = (mse(recon_x, x) * (1 + k * weighting_mask)).sum(dim=-1).mean()

        # KL divergence
        # TODO: act on the prior distribution if you want to embed in the model the 
        # representation score correction
        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = kl_divergence(
            Normal(z1_mean, torch.sqrt(z1_var)),                       # posterior N(z_mean, z_var)
            Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var)) # prior N(0,1)
        ).sum(dim=1).mean()

        if self.use_hsic:
            if not self.hsic_one_vs_all:
                z_ann = z1[:, :-self.n_ext_decoder]
                z_ext = z1[:, -self.n_ext_decoder:]
                hsic_loss = hsic(z_ann, z_ext)
            else:
                hsic_loss = 0.
                sz = self.latent_dim + self.n_ext_encoder
                shift = self.latent_dim + self.n_ext_m_decoder
                for i in range(self.n_ext_decoder):
                    sel_cols = torch.full((sz,), True, device=z1.device)
                    sel_cols[shift + i] = False
                    rest = z1[:, sel_cols]
                    term = z1[:, ~sel_cols]
                    hsic_loss = hsic_loss + hsic(term, rest)
        else:
            hsic_loss = torch.tensor(0.0, device=z1.device)

        return recon_loss, weighted_recon_loss, kl_div, hsic_loss