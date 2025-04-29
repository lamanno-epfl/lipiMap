import scanpy as sc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
from tqdm import tqdm

import math

def visualize_data(adata):
    # fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig, axes = plt.subplots(6, 5, figsize=(40, 60))
    axes = axes.flatten()
    dot_size = 0.7

    # sections_to_plot = [1,5,9,12,15,18,26,31]
    sections_to_plot = [i for i in range(1, 31)]

    global_min_z = adata.obs["z_index"].min()
    global_max_z = adata.obs["z_index"].max()
    global_min_y = -adata.obs["y_index"].max()
    global_max_y = -adata.obs["y_index"].min()

    for i, section_num in enumerate(sections_to_plot):
        ax = axes[i]
        xx = adata.obs[adata.obs["Section"] == section_num]
        sc = ax.scatter(xx["z_index"], -xx["y_index"], c=xx["lipotype_color"], s=dot_size, alpha=1)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)
        ax.set_title(f"Section {section_num}", fontsize=20)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# def visualize_lipids(adata, sorted_indices, section):

#     zz = adata.obs["z_index"]
#     yy = -adata.obs["y_index"]

#     lipids_names = adata.var_names

#     global_min_z = zz.min()
#     global_max_z = zz.max()
#     global_min_y = yy.min()
#     global_max_y = yy.max()

#     fig, axes = plt.subplots(6, 5, figsize=(30, 40))
#     axes = axes.flatten()
#     fig.suptitle(f"Section {section}", size=25)
#     dot_size = 0.7

#     # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
#     cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

#     for i in range(30):
#         ax = axes[i]
#         lipid_values = adata.X[:, sorted_indices[i]]
#         sc = ax.scatter(zz, yy, c=lipid_values, s=dot_size, alpha=1, cmap=cmap)
#         ax.axis("off")
#         ax.set_aspect("equal")
#         ax.set_xlim(global_min_z, global_max_z)
#         ax.set_ylim(global_min_y, global_max_y)
#         ax.set_title(f"{lipids_names[sorted_indices[i]]}", size=15)

#         fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
#     plt.tight_layout()
#     plt.show()


def visualize_pcs(adata, section):

    coord = adata.obs[["Section", "z_index", "y_index"]]
    pcs_list = [f"PCa{i+1}" for i in range(30)]
    pcs = adata.obs[pcs_list]

    global_min_z = coord["z_index"].min()
    global_max_z = coord["z_index"].max()
    global_min_y = -coord["y_index"].max()
    global_max_y = -coord["y_index"].min()

    fig, axes = plt.subplots(6, 5, figsize=(30, 40))
    fig.suptitle(f"Section {section}", size=25)
    dot_size = 0.7

    # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
    cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

    for i, pc in enumerate(pcs):
        ax = axes[i]
        xx = coord[coord["Section"] == section][["z_index", "y_index"]]
        pcs_sec = pcs[coord["Section"] == section]
        sc = ax.scatter(
            xx["z_index"], -xx["y_index"], s=dot_size, alpha=1, c=pcs_sec[pc], cmap=cmap
        )
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)
        ax.set_title(f"PC {i+1}", size=15)

        fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
    plt.tight_layout()
    plt.show()


def umap_latent_space(adata, use_rep="X_cvae", **kwargs):

    # color = [color] if isinstance(color, str) else color

    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, frameon=False, **kwargs)


def visualize_LPs(adata, mask):

    fig, axes = plt.subplots(2, 4, figsize=(25, 15))
    axes = axes.flatten()
    # fig.suptitle(f"Section {section}", size=25)
    dot_size = 0.7

    zz = adata.obs["z_index"]
    yy = -adata.obs["y_index"]

    global_min_z = zz.min()
    global_max_z = zz.max()
    global_min_y = yy.min()
    global_max_y = yy.max()

    variances = np.var(adata.obsm["X_cvae"], axis=0)
    indices_sorted = np.argsort(variances)[::-1]
    lipids_values_sorted = adata.obsm["X_cvae"][:, indices_sorted]

    names_sorted = [adata.uns[mask][i] for i in indices_sorted]

    # cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

    for i, lp in enumerate(names_sorted):
        ax = axes[i]
        lp_values = lipids_values_sorted[:, i]
        sc = ax.scatter(zz, yy, c=lp_values, s=dot_size, alpha=1)

        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)
        ax.set_title(f"LP: {lp}", size=15)

        fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


# def visualize_celltype(adata, name):

#     var_names = adata.var_names
#     if var_names[0].startswith("norm_exp_"):
#         var_names = [var[9:] for var in var_names]

#     if var_names[0].startswith("exp_"):
#         var_names = [var[4:] for var in var_names]

#     fig = plt.figure(figsize=(25, 20))
#     gs = fig.add_gridspec(4, 4)

#     ax0 = fig.add_subplot(gs[1:3, 1:3])

#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax4 = fig.add_subplot(gs[0, 3])

#     ax5 = fig.add_subplot(gs[1, 0])
#     ax6 = fig.add_subplot(gs[1, 3])

#     ax7 = fig.add_subplot(gs[2, 0])
#     ax8 = fig.add_subplot(gs[2, 3])

#     ax9 = fig.add_subplot(gs[3, 0])
#     ax10 = fig.add_subplot(gs[3, 1])
#     ax11 = fig.add_subplot(gs[3, 2])
#     ax12 = fig.add_subplot(gs[3, 3])

#     fig.suptitle(f"Celltype: {name}", size=25)

#     axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

#     cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

#     zz = adata.obs["z_index"]
#     yy = -adata.obs["y_index"]

#     lp_index = adata.uns["celltypes"].index(name.replace(" ", "_"))
#     lp_values = adata.obsm["X_cvae"][:, lp_index]  # np array n_pixels x n_latent_dim

#     lipids_in_lp_indices = np.where(adata.varm["celltypes"][:, lp_index] == 1)[0]

#     # means = adata.X[:, lipids_in_lp_indices].mean(axis=0)
#     # lipids_in_lp_indices_sorted = np.argsort(means)[::-1]
#     # lipids_values_sorted = adata.X[:, lipids_in_lp_indices_sorted]

#     variances = np.var(adata.X[:, lipids_in_lp_indices], axis=0)
#     sorted = np.argsort(variances)[::-1]
#     lipids_values_sorted = adata.X[:, lipids_in_lp_indices[sorted]]

#     var_names = [var_names[i] for i in lipids_in_lp_indices[sorted]]

#     global_min_z = zz.min()
#     global_max_z = zz.max()
#     global_min_y = yy.min()
#     global_max_y = yy.max()

#     for i, ax in enumerate(axes):
#         ax = axes[i]
#         if i == 0:
#             values = lp_values
#             dot_size = 7
#         else:
#             values = lipids_values_sorted[:, i - 1]
#             ax.set_title(f"{i}: {var_names[i-1]}")
#             dot_size = 1

#         sc = ax.scatter(zz, yy, c=values, s=dot_size, alpha=1, cmap=cmap)
#         ax.axis("off")
#         ax.set_aspect("equal")
#         ax.set_xlim(global_min_z, global_max_z)
#         ax.set_ylim(global_min_y, global_max_y)

#         fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
#     plt.tight_layout()
#     plt.show()


# def visualize_reactionfamily(adata, name):

#     var_names = adata.var_names  # [correct_name(adata.var_names[i]) for i in range(adata.n_vars)]
#     if var_names[0].startswith("norm_exp_"):
#         var_names = [var[9:] for var in var_names]

#     if var_names[0].startswith("exp_"):
#         var_names = [var[4:] for var in var_names]

#     fig = plt.figure(figsize=(25, 20))
#     gs = fig.add_gridspec(4, 4)

#     ax0 = fig.add_subplot(gs[1:3, 1:3])

#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])
#     ax4 = fig.add_subplot(gs[0, 3])

#     ax5 = fig.add_subplot(gs[1, 0])
#     ax6 = fig.add_subplot(gs[1, 3])

#     ax7 = fig.add_subplot(gs[2, 0])
#     ax8 = fig.add_subplot(gs[2, 3])

#     ax9 = fig.add_subplot(gs[3, 0])
#     ax10 = fig.add_subplot(gs[3, 1])
#     ax11 = fig.add_subplot(gs[3, 2])
#     ax12 = fig.add_subplot(gs[3, 3])

#     fig.suptitle(f"Reaction Family: {name}", size=25)

#     axes = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

#     cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

#     zz = adata.obs["z_index"]
#     yy = -adata.obs["y_index"]

#     lp_index = adata.uns["reactionfamilies"].index(name.replace(" ", "_"))
#     lp_values = adata.obsm["X_cvae"][:, lp_index]  # np array n_pixels x n_latent_dim

#     lipids_in_lp_indices = np.where(adata.varm["reactionfamilies"][:, lp_index] == 1)[0]

#     # means = adata.X[:, lipids_in_lp_indices].mean(axis=0)
#     # lipids_in_lp_indices_sorted = np.argsort(means)[::-1]
#     # lipids_values_sorted = adata.X[:, lipids_in_lp_indices_sorted]

#     variances = np.var(adata.X[:, lipids_in_lp_indices], axis=0)
#     sorted = np.argsort(variances)[::-1]
#     lipids_values_sorted = adata.X[:, lipids_in_lp_indices[sorted]]

#     var_names = [var_names[i] for i in lipids_in_lp_indices[sorted]]

#     global_min_z = zz.min()
#     global_max_z = zz.max()
#     global_min_y = yy.min()
#     global_max_y = yy.max()

#     for i, ax in enumerate(axes):
#         ax = axes[i]
#         if i == 0:
#             values = lp_values
#             dot_size = 7
#         else:
#             values = lipids_values_sorted[:, i - 1]
#             ax.set_title(f"{i}: {var_names[i-1]}")
#             dot_size = 1

#         sc = ax.scatter(zz, yy, c=values, s=dot_size, alpha=1, cmap=cmap)
#         ax.axis("off")
#         ax.set_aspect("equal")
#         ax.set_xlim(global_min_z, global_max_z)
#         ax.set_ylim(global_min_y, global_max_y)

#         fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
#     plt.tight_layout()
#     plt.show()


def visualize_errors(loss_dict, section=15):
    # loss_dict = dict(logs)
    assert (
        len(loss_dict.keys()) % 2 == 0
    ), "The number of logs must be even, as they represent the training and validation losses"
    n_losses = int(len(loss_dict.keys()) / 2)

    fig, axs = plt.subplots(n_losses, 1, figsize=(20, 15))
    # for key, value in loss_dict.items():
    #     print(key)
    #     print(value)
    loss_list = list(loss_dict.keys())
    loss_types = [loss_list[i][6:] for i in range(n_losses)]

    for i, ax in enumerate(axs):
        ax.semilogy(loss_dict[f"epoch_{loss_types[i]}"], label=f"train_{loss_types[i]}")
        ax.semilogy(loss_dict[f"val_{loss_types[i]}"], label=f"val_{loss_types[i]}")
        ax.legend()
        ax.set_title(f"{loss_types[i]}", size=15)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")

    # Show the plot
    plt.tight_layout()
    plt.show()


def visualize_single_residuals(adata_orig, adata_recon, lipid):

    zz = adata_orig.obs["z_index"]
    yy = -adata_orig.obs["y_index"]

    global_min_z = zz.min()
    global_max_z = zz.max()
    global_min_y = yy.min()
    global_max_y = yy.max()

    lipids_names = list(adata_orig.var_names)
    if lipids_names[0].startswith("norm_exp_"):
        lipids_names = [var[9:] for var in lipids_names]

    if lipids_names[0].startswith("exp_"):
        lipids_names = [var[4:] for var in lipids_names]

    # if lipid=='total':
    fig, axes = plt.subplots(1, 3, figsize=(25, 9))
    axes = axes.flatten()
    fig.suptitle(f"Lipid {lipid}", size=25)
    dot_size = 3

    orig = adata_orig.X[:, lipids_names.index(lipid)]
    recon = adata_recon[:, lipids_names.index(lipid)]

    # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
    # cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

    vmin = min(orig.min(), recon.min())
    vmax = max(orig.max(), recon.max())
    # norm = Normalize(vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=-10, vmax=100)

    rel_err = orig - recon
    vmax_abs_rel_err = np.max(np.abs(rel_err))
    norm_rel_err = Normalize(vmin=-vmax_abs_rel_err, vmax=vmax_abs_rel_err)
    norm_rel_err = Normalize(vmin=-60, vmax=60)

    sc0 = axes[0].scatter(zz, yy, c=orig, s=dot_size, alpha=1, cmap="viridis", norm=norm)
    axes[0].set_title("Original", size=15)
    sc1 = axes[1].scatter(zz, yy, c=recon, s=dot_size, alpha=1, cmap="viridis", norm=norm)
    axes[1].set_title("Reconstructed", size=15)
    sc2 = axes[2].scatter(zz, yy, c=rel_err, s=dot_size, alpha=1, cmap="RdBu", norm=norm_rel_err)
    axes[2].set_title("Relative Error", size=15)

    for ax in axes:
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)

    cbar_ax1 = fig.add_axes([0.01, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    fig.colorbar(sc0, cax=cbar_ax1, orientation="horizontal")

    cbar_ax2 = fig.add_axes([0.67, 0.1, 0.32, 0.03])  # Adjust these values as needed
    fig.colorbar(sc2, cax=cbar_ax2, orientation="horizontal")

    # fig.colorbar(sc0, ax=ax, pad=0.01, shrink=0.6)

    # for j in range(i+1, len(axes)):
    #     fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
    plt.tight_layout()
    plt.show()


# def visualize_multiple_residuals(adata_orig, adata_recon, names_list):

#     zz = adata_orig.obs["z_index"]
#     yy = -adata_orig.obs["y_index"]

#     lipids_names = adata_orig.var_names
#     if lipids_names[0].startswith("norm_exp_"):
#         lipids_names = [var[9:] for var in lipids_names]

#     if lipids_names[0].startswith("exp_"):
#         lipids_names = [var[4:] for var in lipids_names]

#     global_min_z = zz.min()
#     global_max_z = zz.max()
#     global_min_y = yy.min()
#     global_max_y = yy.max()

#     fig, axes = plt.subplots(len(names_list), 3, figsize=(25, 60))
#     # fig.suptitle(f"Section {section}", size=25)
#     dot_size = 0.7
#     # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
#     cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

#     vmin = adata_orig.X.min()
#     vmax = adata_orig.X.max()
#     vmax_abs_rel_err = 0

#     l_indexes = [lipids_names.index(name.replace(" ", "_")) for name in names_list]

#     for i in range(len(names_list)):

#         orig = adata_orig.X[:, l_indexes[i]]
#         recon = adata_recon[:, l_indexes[i]]
#         rel_err = orig - recon
#         vmax_abs_rel_err_curr = np.max(np.abs(rel_err))
#         if vmax_abs_rel_err_curr > vmax_abs_rel_err:
#             vmax_abs_rel_err = vmax_abs_rel_err_curr

#         # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
#         # cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
#         vmin_curr = min(orig.min(), recon.min())
#         if vmin_curr < vmin:
#             vmin = vmin_curr
#         vmax_curr = max(orig.max(), recon.max())
#         if vmax_curr > vmax:
#             vmax = vmax_curr

#     norm_rel_err = Normalize(vmin=-vmax_abs_rel_err, vmax=vmax_abs_rel_err)
#     norm = Normalize(vmin=vmin, vmax=vmax)

#     for i in range(len(names_list)):
#         orig = adata_orig.X[:, l_indexes[i]]
#         recon = adata_recon[:, l_indexes[i]]
#         rel_err = orig - recon

#         sc0 = axes[i][0].scatter(zz, yy, c=orig, s=dot_size, alpha=1, cmap="viridis", norm=norm)
#         axes[i][0].set_title(f"Original {lipids_names[l_indexes[i]]}", size=15)
#         sc1 = axes[i][1].scatter(zz, yy, c=recon, s=dot_size, alpha=1, cmap="viridis", norm=norm)
#         axes[i][1].set_title(f"Reconstructed {lipids_names[l_indexes[i]]}", size=15)
#         sc2 = axes[i][2].scatter(
#             zz, yy, c=rel_err, s=dot_size, alpha=1, cmap="RdBu", norm=norm_rel_err
#         )
#         axes[i][2].set_title(f"Relative Error {lipids_names[l_indexes[i]]}", size=15)

#         for j in range(3):
#             axes[i][j].axis("off")
#             axes[i][j].set_aspect("equal")
#             axes[i][j].set_xlim(global_min_z, global_max_z)
#             axes[i][j].set_ylim(global_min_y, global_max_y)

#         cbar_ax1 = fig.add_axes(
#             [0.01, 0.1 + (i - 1) * 0.1, 0.65, 0.005]
#         )  # [left, bottom, width, height]
#         fig.colorbar(sc0, cax=cbar_ax1, orientation="horizontal")

#         cbar_ax2 = fig.add_axes([0.67, 0.1 + (i - 1) * 0.1, 0.32, 0.005])
#         fig.colorbar(sc2, cax=cbar_ax2, orientation="horizontal")

#         # fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     # plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
#     plt.tight_layout(pad=3)
#     plt.show()


def visualize_program_residuals(adata_orig, adata_recon, mask, name, global_ranking):

    zz = adata_orig.obs["z_index"]
    yy = -adata_orig.obs["y_index"]

    lipids_names = adata_orig.var_names
    if lipids_names[0].startswith("norm_exp_"):
        lipids_names = [var[9:] for var in lipids_names]

    if lipids_names[0].startswith("exp_"):
        lipids_names = [var[4:] for var in lipids_names]

    global_min_z = zz.min()
    global_max_z = zz.max()
    global_min_y = yy.min()
    global_max_y = yy.max()

    lp_index = adata_orig.uns[mask].index(name.replace(" ", "_"))
    # lp_values = adata_orig.obsm['X_cvae'][:, lp_index] # np array n_pixels x n_latent_dim

    # global_ranking is a list of all lipids sorted by variance
    # local_indices is a list of all lipids in the reaction family
    # local_ranking is a list of all lipids in the reaction family sorted according to global_ranking

    local_indices = np.where(adata_orig.varm[mask][:, lp_index] == 1)[0]
    local_ranking = sorted(local_indices, key=lambda x: global_ranking[x])
    # print(lipids_in_lp_indices)

    # means = adata.X[:, lipids_in_lp_indices].mean(axis=0)
    # lipids_in_lp_indices_sorted = np.argsort(means)[::-1]
    # lipids_values_sorted = adata.X[:, lipids_in_lp_indices_sorted]

    # variances = np.var(adata.X[:, lipids_in_lp_indices], axis=0)
    # sorted = np.argsort(variances)[::-1]
    # lipids_values_sorted = adata.X[:, lipids_in_lp_indices[sorted]]

    fig, axes = plt.subplots(10, 3, figsize=(25, 60))
    # fig.suptitle(f"Section {section}", size=25)
    dot_size = 0.7
    # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
    cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

    vmin = adata_orig.X.min()
    vmax = adata_orig.X.max()
    vmax_abs_rel_err = 0

    for i in range(10):
        orig = adata_orig.X[:, local_ranking[i]]
        recon = adata_recon[:, local_ranking[i]]
        rel_err = orig - recon
        vmax_abs_rel_err_curr = np.max(np.abs(rel_err))
        if vmax_abs_rel_err_curr > vmax_abs_rel_err:
            vmax_abs_rel_err = vmax_abs_rel_err_curr

        # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
        # cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
        vmin_curr = min(orig.min(), recon.min())
        if vmin_curr < vmin:
            vmin = vmin_curr
        vmax_curr = max(orig.max(), recon.max())
        if vmax_curr > vmax:
            vmax = vmax_curr

    norm_rel_err = Normalize(vmin=-vmax_abs_rel_err, vmax=vmax_abs_rel_err)
    # norm = Normalize(vmin=vmin, vmax=vmax)
    norm = Normalize(vmin=0, vmax=vmax)

    for i in range(10):
        orig = adata_orig.X[:, local_ranking[i]]
        recon = adata_recon[:, local_ranking[i]]
        rel_err = orig - recon

        sc0 = axes[i][0].scatter(zz, yy, c=orig, s=dot_size, alpha=1, cmap="viridis", norm=norm)
        axes[i][0].set_title(f"Original {lipids_names[local_ranking[i]]}", size=15)
        sc1 = axes[i][1].scatter(zz, yy, c=recon, s=dot_size, alpha=1, cmap="viridis", norm=norm)
        axes[i][1].set_title(f"Reconstructed {lipids_names[local_ranking[i]]}", size=15)
        sc2 = axes[i][2].scatter(
            zz, yy, c=rel_err, s=dot_size, alpha=1, cmap="RdBu", norm=norm_rel_err
        )
        axes[i][2].set_title(f"Relative Error {lipids_names[local_ranking[i]]}", size=15)

        for j in range(3):
            axes[i][j].axis("off")
            axes[i][j].set_aspect("equal")
            axes[i][j].set_xlim(global_min_z, global_max_z)
            axes[i][j].set_ylim(global_min_y, global_max_y)

        cbar_ax1 = fig.add_axes(
            [0.01, 0.1 + (i - 1) * 0.1, 0.65, 0.004]
        )  # [left, bottom, width, height]
        fig.colorbar(sc0, cax=cbar_ax1, orientation="horizontal")

        cbar_ax2 = fig.add_axes([0.67, 0.1 + (i - 1) * 0.1, 0.32, 0.004])
        fig.colorbar(sc2, cax=cbar_ax2, orientation="horizontal")

        # fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
    plt.tight_layout(pad=3)
    plt.show()


# def visualize_residuals(adata, section=15):

#     zz = adata.obs['z_index']
#     yy = -adata.obs['y_index']

#     lipids_names = adata.var_names

#     global_min_z = zz.min()
#     global_max_z = zz.max()
#     global_min_y = yy.min()
#     global_max_y = yy.max()

#     fig, axes = plt.subplots(6, 5, figsize=(30, 40))
#     axes = axes.flatten()
#     fig.suptitle(f"Section {section}", size=25)
#     dot_size = 0.7

#     # Create a colormap where the minimum corresponds to blue and the maximum to red, going through white
#     cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])

#     for i in range(30):
#         ax = axes[i]
#         error = adata.X[:, i]
#         sc = ax.scatter(zz, yy, c=error, s=dot_size, alpha=1, cmap=cmap)
#         ax.axis('off')
#         ax.set_aspect('equal')
#         ax.set_xlim(global_min_z, global_max_z)
#         ax.set_ylim(global_min_y, global_max_y)
#         ax.set_title(f"{lipids_names[i]}", size=15)

#         fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.6)

#     for j in range(i+1, len(axes)):
#         fig.delaxes(axes[j])

#     plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
#     plt.tight_layout()
#     plt.show()


def visualize_progession_conditioning():
    pass


def visualize_reconstruction_evolution(adata_orig, adata_recon_list, lipid, path=None):

    zz = adata_orig.obs["z_index"]
    yy = -adata_orig.obs["y_index"]

    global_min_z = zz.min()
    global_max_z = zz.max()
    global_min_y = yy.min()
    global_max_y = yy.max()

    lipids_names = list(adata_orig.var_names)
    # print(lipids_names)

    orig = adata_orig.X[:, lipids_names.index(lipid)]
    vmax_abs_rel_err = 0
    vmin = orig.min()
    vmax = orig.max()

    for adata_recon in adata_recon_list:
        recon = adata_recon[:, lipids_names.index(lipid)]
        rel_err = orig - recon
        vmax_abs_rel_err_curr = np.max(np.abs(rel_err))
        if vmax_abs_rel_err_curr > vmax_abs_rel_err:
            vmax_abs_rel_err = vmax_abs_rel_err_curr

        vmin_curr = min(orig.min(), recon.min())
        vmax_curr = max(orig.max(), recon.max())
        if vmin_curr < vmin:
            vmin = vmin_curr
        if vmax_curr > vmax:
            vmax = vmax_curr

    norm_rel_err = Normalize(vmin=-vmax_abs_rel_err, vmax=vmax_abs_rel_err)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, adata_recon in enumerate(adata_recon_list):
        fig, axes = plt.subplots(1, 3, figsize=(25, 9))
        axes = axes.flatten()
        fig.suptitle(f"Lipid {lipid}", size=25)
        dot_size = 3
        recon = adata_recon[:, lipids_names.index(lipid)]
        rel_err = orig - recon

        sc0 = axes[0].scatter(zz, yy, c=orig, s=dot_size, alpha=1, cmap="viridis", norm=norm)
        axes[0].set_title("Original", size=15)
        sc1 = axes[1].scatter(zz, yy, c=recon, s=dot_size, alpha=1, cmap="viridis", norm=norm)
        axes[1].set_title("Reconstructed", size=15)
        sc2 = axes[2].scatter(
            zz, yy, c=rel_err, s=dot_size, alpha=1, cmap="RdBu", norm=norm_rel_err
        )
        axes[2].set_title("Relative Error", size=15)

        for ax in axes:
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(global_min_z, global_max_z)
            ax.set_ylim(global_min_y, global_max_y)

        cbar_ax1 = fig.add_axes([0.01, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
        fig.colorbar(sc0, cax=cbar_ax1, orientation="horizontal")

        cbar_ax2 = fig.add_axes([0.67, 0.1, 0.32, 0.03])  # Adjust these values as needed
        fig.colorbar(sc2, cax=cbar_ax2, orientation="horizontal")

        plt.subplots_adjust(wspace=0, hspace=0)  # adjust the padding
        plt.tight_layout()
        if path:
            plt.savefig(f"{path}/lipid_{lipid}_reconstruction_{i}.png")
        else:
            plt.show()

    # # Get all files in the directory
    # images = [img for img in os.listdir(path) if img.endswith(".png")]
    # # Sort files alphabetically
    # images.sort()

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID'
    # out = cv2.VideoWriter(path, fourcc, fps=5)

    # for image in images:
    #     img_path = os.path.join(path, image)
    #     frame = cv2.imread(img_path)
    #     # Resize frame to match the specified frame size
    #     #frame = cv2.resize(frame, frame_size)
    #     # Write the frame into the file 'output_video.mp4'
    #     out.write(frame)

    # # Release everything when job is finished
    # out.release()
    # print("The video was successfully created.")


def plot_decoder_weights(model, adata, mask_framework, alpha_l1=None):
    decoder_weights = model.model.decoder.L0.expr_L.weight.data.cpu().numpy()
    # print(decoder_weights.shape)
    # df_decoder_weights = pd.DataFrame(decoder_weights, columns=adata.uns[mask_framework], index=adata.var_names)
    # df_decoder_weights.to_csv('/home/fventuri/francesca/df_decoder_weights.csv')
    mask = model.model.mask.t()

    if not model.soft_mask_:
        decoder_weights[np.where(mask == 0)] = (
            np.nan
        )  # Setting to NaN will make them white in the heatmap

    vmin = np.abs(np.nanmin(decoder_weights))  # Use nanmin to ignore NaNs in calculations
    vmax = np.abs(np.nanmax(decoder_weights))  # Use nanmax to ignore NaNs in calculations

    # Create heatmap with PuOr colormap
    fig, ax1 = plt.subplots(figsize=(35, 35))
    sns.heatmap(
        decoder_weights,
        cmap="PuRd" if model.soft_mask_ else "viridis",
        ax=ax1,
        # cbar_kws={'label': 'Enrichment'},
        xticklabels=adata.uns[mask_framework],
        yticklabels=adata.var_names,
        vmin=vmin,
        vmax=vmax,
        mask=np.isnan(decoder_weights),  # Mask NaN values to show as white
    )

    # Customize ticks
    ax1.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax1.tick_params(axis="y", which="both", left=False, right=False, pad=20)
    ax1.set_title(
        (
            f"Decoder weights - Mask Membership: SOFT - ALPHA L1: {alpha_l1}"
            if model.soft_mask_
            else "Decoder weights - Mask Membership: HARD"
        ),
        fontsize=20,
    )

    # Display the heatmap
    plt.show()


def pdf_spatial_representation(adata, what_data, output_file, mask_framework="all"):

    assert (
        what_data == "X_latent" or what_data == "X_reconstruction" or what_data == "X_residuals"
    ), "what_data must be either 'X_latent' or 'X_reconstruction' or 'X_residuals'"
    
    coords = adata.obs[["Section", "x", "y"]]  # 'zccf', 'yccf', 'z_index', 'y_index',

    pdf_pages = PdfPages(output_file)

    elements = adata.uns[mask_framework] if what_data == "X_latent" else adata.var_names
    
    for i, curr in tqdm(enumerate(elements)):

        results = []
        filtered_data = pd.concat(
            [
                coords,
                pd.DataFrame(adata.obsm[what_data][:, i], index=adata.obs_names, columns=[curr]),
            ],
            axis=1,
        )

        sections = filtered_data["Section"].unique()
        for section in sections:
            subset = filtered_data[filtered_data["Section"] == section]
            perc_2 = subset[curr].quantile(0.02)
            perc_98 = subset[curr].quantile(0.98)
            results.append([section, perc_2, perc_98])

        percentile_df = pd.DataFrame(results, columns=["Section", "2-perc", "98-perc"])
        med2p = percentile_df["2-perc"].median()
        med98p = percentile_df["98-perc"].median()
        cmap = plt.cm.PuOr
        
        num_cols = min(len(sections), 8)
        num_rows = math.ceil(len(sections) / num_cols)

        # Create the subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 3*num_rows), rasterized=True)

        axes = axes.flatten()
        for j, section in enumerate(sorted(sections)):
            ax = axes[j]
            ddf = filtered_data[(filtered_data["Section"] == section)]
            ax.scatter(
                ddf["y"], # 'zccf'
                -ddf["x"], # 'yccf
                c=ddf[curr],
                cmap="PuOr",
                s=0.5,
                rasterized=True,
                vmin=med2p,
                vmax=med98p,
            )

        for ax in axes:
            ax.axis("off")
            ax.set_aspect("equal")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=med2p, vmax=med98p)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(curr)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        pdf_pages.savefig(fig)
        plt.close(fig)

    pdf_pages.close()
