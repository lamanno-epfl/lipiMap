import numpy as np


def ensure_posdef(matrix):
    """
    Ensures that a given square matrix is positive definite by adjusting its eigenvalues.

    Parameters
    ----------
    matrix : numpy.ndarray
        A square numpy matrix which is to be adjusted to ensure it is positive definite.

    Returns
    -------
    numpy.ndarray
        A positive definite matrix derived from the input matrix by adjusting its eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues[eigenvalues <= 0] = (
        np.min(eigenvalues[eigenvalues > 0]) * 1e-5
    )
    pd_matrix = np.dot(
        eigenvectors, np.dot(np.diag(eigenvalues), eigenvectors.T)
    )

    return pd_matrix


def compute_adjusting_weight_cov_matrix(cov_matrix, mask):
    """
    Computes an adjusted weight covariance matrix based on a given binary mask indicating lipid program membership.

    Parameters
    ----------
    cov_matrix : numpy.ndarray
        The original covariance matrix of lipids, assumed to be square (i.e., dimensions are N x N where N is the number of lipids).
    mask : numpy.ndarray
        A binary mask array of shape (N, M) where N is the number of lipids and M is the number of lipid programs.
        A value of 1 in position (i, j) indicates that lipid i is a member of program j.

    Returns
    -------
    numpy.ndarray
        The adjusted weight covariance matrix, where weights are increased for each pair of lipids belonging to the same lipid program.
    """
    weight_matrix = np.ones_like(cov_matrix)

    for j in range(mask.shape[1]):
        lipids_in_program = np.where(mask[:, j] == 1)[0]

        for i in lipids_in_program:
            for k in lipids_in_program:
                if i != k:
                    weight_matrix[i, k] += 1
                    weight_matrix[k, i] += 1

    return weight_matrix
