import torch
from torch.autograd import Variable

from ._utils import partition

# Adapted from
# Title: Biologically informed deep learning to query gene programs in single-cell atlases
# Authors: Mohammad Lotfollahi, Sergei Rybakov, Karin Hrovatin, Soroor Hediyeh-zadeh,
#          Carlos Talavera-López, Alexander V. Misharin & Fabian J. Theis
# Code: https://github.com/theislab/scarches/tree/master/scarches/models/trvae/losses.py


def mse(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data.

    Parameters
    ----------
    recon_x: torch.Tensor
         Torch Tensor of reconstructed data
    x: torch.Tensor
         Torch Tensor of ground truth data

    Returns
    -------
    MSE loss value
    """
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="none")
    return mse_loss


def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

    Parameters
    ----------
    x: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    y: torch.Tensor
         Tensor with shape [batch_size, z_dim].
    alphas: Tensor

    Returns
    -------
    Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1.0 / (2.0 * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

    - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

    Parameters
    ----------
    source_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]
    target_features: torch.Tensor
         Tensor with shape [batch_size, z_dim]

    Returns
    -------
    Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        100,
        1e3,
        1e4,
        1e5,
        1e6,
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(
        device=source_features.device
    )

    cost = torch.mean(
        gaussian_kernel_matrix(source_features, source_features, alphas)
    )
    cost += torch.mean(
        gaussian_kernel_matrix(target_features, target_features, alphas)
    )
    cost -= 2 * torch.mean(
        gaussian_kernel_matrix(source_features, target_features, alphas)
    )

    return cost


def mmd(y, c, n_conditions, beta, boundary):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

    Parameters
    ----------
    n_conditions: integer
         Number of classes (conditions) the data contain.
    beta: float
         beta coefficient for MMD loss.
    boundary: integer
         If not 'None', mmd loss is only calculated on #new conditions.
    y: torch.Tensor
         Torch Tensor of computed latent data.
    c: torch.Tensor
         Torch Tensor of condition labels.

    Returns
    -------
    Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = partition(y, c, n_conditions)
    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            for j in range(boundary, n_conditions):
                if (
                    conditions_mmd[i].size(0) < 2
                    or conditions_mmd[j].size(0) < 2
                ):
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i, cond in enumerate(conditions_mmd):
            if cond.size(0) < 1:
                continue
            for j in range(i):
                if cond.size(0) < 1 or i == j:
                    continue
                loss += mmd_loss_calc(cond, conditions_mmd[j])

    return beta * loss
