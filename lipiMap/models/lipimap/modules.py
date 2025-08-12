import torch
import torch.nn as nn

import numpy as np

from typing import Optional

from ..trvae._utils import one_hot_encoder

# Adapted from
# Title: Biologically informed deep learning to query gene programs in single-cell atlases
# Authors: Mohammad Lotfollahi, Sergei Rybakov, Karin Hrovatin, Soroor Hediyeh-zadeh, Carlos Talavera-López, Alexander V. Misharin & Fabian J. Theis
# Code: https://github.com/theislab/scarches/tree/master/scarches/models/expimap/modules.py


class ZeroOneClipper(object):
    def __call__(self, module):
        """
        Clips the weights of the given module to the range [0, 1].

        Parameters
        ----------
        module : torch.nn.Module
            The neural network module whose weights are to be clipped.
        """
        if hasattr(module, "weight"):
            with torch.no_grad():
                w = module.weight.data
                w.clamp_(0.0, 1.0)  # 0 <= F_ij <= 1


class MaskedLinear(nn.Linear):
    """
    A linear layer with a mask applied to the weights.

    Parameters
    ----------
    n_in : int
        The number of input features.
    n_out : int
        The number of output features.
    mask : torch.Tensor
        A binary mask of shape (n_in, n_out) indicating the connections to preserve.
    bias : bool, optional (default=True)
        Whether to include a bias term.
    """

    def __init__(self, n_in, n_out, mask, bias=True):
        if n_in != mask.shape[0] or n_out != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask.")

        super().__init__(n_in, n_out, bias)

        # The mask is stored as a buffer to avoid optimization.
        self.register_buffer("mask", mask.t())

        self.weight.data *= self.mask

    def forward(self, input):
        return nn.functional.linear(
            input, self.weight * self.mask, self.bias
        )


class MaskedCondLayers(nn.Module):
    """
    A modular layer that combines masked linear transformations
    with conditional inputs and external nodes.

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    n_cond : int
        Number of conditional features.
    bias : bool
        Whether to include a bias term in the main layer.
    n_ext : int, optional (default=0)
        Number of unconstrained external features.
    n_ext_m : int, optional (default=0)
        Number of constrained external features.
    mask : torch.Tensor, optional (default=None)
        Binary mask for the main linear layer.
    ext_mask : torch.Tensor, optional (default=None)
        Binary mask for the constrained external layer.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cond: int,
        bias: bool,
        n_ext: int = 0,
        n_ext_m: int = 0,
        mask: Optional[torch.Tensor] = None,
        ext_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        # For the ordinary nodes
        if mask is None:
            self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        else:
            self.expr_L = MaskedLinear(n_in, n_out, mask, bias=bias)

        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

        # For additional external nodes (Architecture Surgery)
        if self.n_ext != 0:
            self.ext_L = nn.Linear(self.n_ext, n_out, bias=False)

        if self.n_ext_m != 0:
            if ext_mask is not None:
                self.ext_L_m = MaskedLinear(
                    self.n_ext_m, n_out, ext_mask, bias=False
                )
            else:
                self.ext_L_m = nn.Linear(self.n_ext_m, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Performs a forward pass, combining multiple input sources.
        """

        if self.n_cond == 0:
            expr, cond = x, None
        else:
            expr, cond = torch.split(
                x, [x.shape[1] - self.n_cond, self.n_cond], dim=1
            )

        if self.n_ext == 0:
            ext = None
        else:
            expr, ext = torch.split(
                expr, [expr.shape[1] - self.n_ext, self.n_ext], dim=1
            )

        if self.n_ext_m == 0:
            ext_m = None
        else:
            expr, ext_m = torch.split(
                expr, [expr.shape[1] - self.n_ext_m, self.n_ext_m], dim=1
            )

        out = self.expr_L(expr)
        if ext is not None:
            out = out + self.ext_L(ext)
        if ext_m is not None:
            out = out + self.ext_L_m(ext_m)
        if cond is not None:
            out = out + self.cond_L(cond)
        return out


class MaskedLinearDecoder(nn.Module):
    """
    A decoder architecture with a masked linear layer and optional additional inputs.

    Parameters
    ----------
    in_dim : int
        Input dimension of the latent space.
    out_dim : int
        Output dimension of the reconstructed space.
    n_cond : int
        Number of conditional features.
    condition_key : str
        Key identifying the type of condition.
    mask : torch.Tensor
        Binary mask for the main layer.
    ext_mask : torch.Tensor
        Binary mask for the external constrained layer.
    recon_loss : str
        Type of reconstruction loss ("mse" is supported).
    last_layer : str, optional
        The final transformation layer ("identity" is supported).
    n_ext : int, optional (default=0)
        Number of unconstrained external features.
    n_ext_m : int, optional (default=0)
        Number of constrained external features.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        n_cond,
        condition_key,
        mask,
        ext_mask,
        recon_loss,
        last_layer=None,
        n_ext=0,
        n_ext_m=0,
    ):
        super().__init__()

        if recon_loss == "mse":
            last_layer = "identity" if last_layer is None else last_layer
        else:
            raise ValueError("Unrecognized loss.")

        print("Decoder Architecture:")
        print(
            f"\tMasked linear layer : {in_dim} (in) + {n_cond} (cond) + {n_ext_m} (ext constr) + {n_ext} (ext unconstr) --> {out_dim} (out)"
        )
        if mask is not None:
            print("\twith hard mask.")
        else:
            print("\twith soft mask.")

        self.n_ext = n_ext
        self.n_ext_m = n_ext_m

        self.n_cond = 0
        if n_cond is not None:
            self.n_cond = n_cond
        self.condition_key = condition_key

        self.L0 = MaskedCondLayers(
            in_dim,
            out_dim,
            n_cond,
            bias=False,
            n_ext=n_ext,
            n_ext_m=n_ext_m,
            mask=mask,
            ext_mask=ext_mask,
        )

        if last_layer == "identity":
            self.mean_decoder = lambda a: a
        else:
            raise ValueError("Unrecognized last layer.")

        print("Last Decoder layer:", last_layer)

    def forward(self, z, batch=None):
        """
        Decodes the latent space into the reconstructed space.

        Parameters
        ----------
        z : torch.Tensor
            Latent space tensor of shape (batch_size, in_dim).
        batch : torch.Tensor, optional
            Batch assignment tensor for conditional inputs.

        Returns
        -------
        tuple
            - `recon_x` (torch.Tensor): Reconstructed output of shape (batch_size, out_dim).
            - `dec_latent` (torch.Tensor): Decoder output before the final transformation.
        """

        if batch is not None:
            if self.condition_key != "arbitrary":
                batch = one_hot_encoder(batch, n_cls=self.n_cond)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.L0(z_cat)
        else:
            dec_latent = self.L0(z)

        recon_x = self.mean_decoder(dec_latent)

        return recon_x, dec_latent

    def nonzero_terms(self):
        """
        Identifies active terms in the decoder's weight matrix.
        """
        v = self.L0.expr_L.weight.data
        nz = (v.norm(p=1, dim=0) > 0).cpu().numpy()
        nz = np.append(nz, np.full(self.n_ext_m, True))
        nz = np.append(nz, np.full(self.n_ext, True))
        return nz

    def n_inactive_terms(self):
        """
        Counts the number of inactive terms in the decoder's weight matrix.
        """
        n = (~self.nonzero_terms()).sum()
        return int(n)


class ExtEncoder(nn.Module):
    """
    An encoder architecture with optional conditional inputs and latent space expansion.

    Parameters
    ----------
    layer_sizes : list of int
        List of layer sizes for the fully connected architecture.
    latent_dim : int
        Dimension of the latent space.
    use_bn : bool
        Whether to use batch normalization.
    use_ln : bool
        Whether to use layer normalization.
    use_dr : bool
        Whether to use dropout.
    dr_rate : float
        Dropout rate, if `use_dr` is True.
    num_classes : int, optional
        Number of conditional classes.
    condition_key : str, optional
        Key identifying the type of condition.
    n_expand : int, optional (default=0)
        Size of the expanded latent space.
    """

    def __init__(
        self,
        layer_sizes: list,
        latent_dim: int,
        use_bn: bool,
        use_ln: bool,
        use_dr: bool,
        dr_rate: float,
        num_classes: Optional[int] = None,
        condition_key: Optional[str] = None,
        n_expand: int = 0,
    ):
        super().__init__()
        self.n_classes = 0
        self.n_expand = n_expand
        if num_classes is not None:
            self.n_classes = num_classes
        self.condition_key = condition_key
        self.FC = None  # Fully Connected
        if len(layer_sizes) > 1:
            print("Encoder Architecture:")
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(
                zip(layer_sizes[:-1], layer_sizes[1:])
            ):
                if i == 0:
                    print(
                        f"\tInput Layer : {in_size} (in) + {self.n_classes} (cond) --> {out_size} (out)"
                    )
                    self.FC.add_module(
                        name="L{:d}".format(i),
                        module=MaskedCondLayers(
                            in_size, out_size, self.n_classes, bias=True
                        ),
                    )
                else:
                    print(
                        f"\tHidden Layer {i} : {in_size} (in) --> {out_size} (out)"
                    )
                    self.FC.add_module(
                        name="L{:d}".format(i),
                        module=nn.Linear(in_size, out_size, bias=True),
                    )
                if use_bn:
                    self.FC.add_module(
                        "N{:d}".format(i),
                        module=nn.BatchNorm1d(out_size, affine=True),
                    )
                elif use_ln:
                    self.FC.add_module(
                        "N{:d}".format(i),
                        module=nn.LayerNorm(
                            out_size, elementwise_affine=False
                        ),
                    )
                self.FC.add_module(
                    name="A{:d}".format(i), module=nn.ReLU()
                )
                if use_dr:
                    self.FC.add_module(
                        name="D{:d}".format(i),
                        module=nn.Dropout(p=dr_rate),
                    )

        print(
            f"\tMean/Var Layer : {layer_sizes[-1]} (in) --> {latent_dim} (out)"
        )  # Mean/Var Layer for the parameters of the latent space
        self.mean_encoder = nn.Linear(layer_sizes[-1], latent_dim)
        self.log_var_encoder = nn.Linear(layer_sizes[-1], latent_dim)

        if self.n_expand != 0:
            print(
                f"\tExpanded Mean/Var Layer : {layer_sizes[-1]} (in) --> {self.n_expand} (out)"
            )
            self.expand_mean_encoder = nn.Linear(
                layer_sizes[-1], self.n_expand
            )
            self.expand_var_encoder = nn.Linear(
                layer_sizes[-1], self.n_expand
            )

    def forward(self, x, batch=None):
        """
        Encodes the input into a latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).
        batch : torch.Tensor, optional
            Batch assignment tensor for conditional inputs.
        """
        if batch is not None:
            if self.condition_key != "arbitrary":
                batch = one_hot_encoder(batch, n_cls=self.n_classes)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)
        means = self.mean_encoder(x)
        log_vars = self.log_var_encoder(x)

        if self.n_expand != 0:
            means = torch.cat((means, self.expand_mean_encoder(x)), dim=-1)
            log_vars = torch.cat(
                (log_vars, self.expand_var_encoder(x)), dim=-1
            )

        return means, log_vars
