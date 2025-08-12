import numpy as np
import pandas as pd
import torch
import anndata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from ..models import LIPIMAP
from ..utils.kpca import ensure_posdef, compute_adjusting_weight_cov_matrix

from tqdm import tqdm
import math

from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression

import scipy.cluster.hierarchy as sch

torch.set_printoptions(precision=3, sci_mode=False, edgeitems=7)
np.set_printoptions(precision=2, edgeitems=7)


def diagonalize_heatmap_layout(df):
    """
    Reorders and cleans a dataframe and its corresponding mask to optimize heatmap visualization.

    Parameters:
        df (pd.DataFrame): The initial dataframe to be reordered and cleaned.
        mask (np.ndarray): Binary mask corresponding to the dataframe.
        mask_key (str): Key to access specific annotations from `adata.uns`.
        adata (anndata.AnnData): Annotated data matrix containing variable names and uns annotations.

    Returns:
        pd.DataFrame: The reordered and cleaned dataframe.
        np.ndarray: The reordered mask.
    """

    # Perform hierarchical clustering and get the column order
    linkage = sch.linkage(
        sch.distance.pdist(df.T), method="weighted", optimal_ordering=True
    )
    column_order = sch.leaves_list(linkage)
    df = df.iloc[:, column_order]

    # Get the row order based on the maximum values
    row_order = np.argmax(df.values, axis=1)
    row_order = np.argsort(row_order)
    df = df.iloc[row_order, :]

    # Drop NaNs and replace infinities in df_to_diagonalize
    df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    return df, row_order, column_order


class LIPIMAP_EVAL:
    """
    Evaluation class for the LIPIMAP model, providing tools for analyzing
    and visualizing the results of lipidomics data analysis.

    Parameters
    ----------
    model : LIPIMAP
        An instance of the LIPIMAP model.
    adata : anndata.AnnData
        Annotated data matrix used for the analysis.
    trainer : lipiMapTrainer, optional
        Trainer used for training the model (default is None).
    condition_key : str, optional
        Key to access conditional labels within `adata.obs` (default is None).
    """

    def __init__(
        self,
        model: LIPIMAP,
        adata: anndata.AnnData,
        condition_key: str = None,
    ):
        """
        Initializes the LIPIMAP_EVAL class with model, data, and optional trainer and condition key.
        """

        self.model = model
        self.logs = dict(self.model.logs_)

        self.trainer = self.model.trainer

        self.adata = adata
        self._config_spatials()

        self.conditions = self.model.model.conditions
        # if condition_key is not None:
        #     self.conditions = label_encoder(
        #         self.adata,
        #         encoder=model.model.condition_encoder,
        #         condition_key=condition_key,
        #     )
        self.batch_names = None
        if condition_key is not None:
            self.batch_names = adata.obs[condition_key].tolist()
        self.adata.obsm["latent"] = self.model.get_latent(
            self.adata.X,
            c=self.conditions,
        )
        self.adata.obsm["output"] = (
            self.model.get_y(
                self.adata.X,
                c=self.conditions,
            )
            .round()
            .astype(int)
        )
        self.adata.obsm["residual"] = (
            self.adata.X - self.adata.obsm["output"]
        )

    def _config_spatials(self):
        """
        Configures spatial parameters from `adata` needed for further spatial analysis.
        """
        self.section = self.adata.obs["SectionID"]
        self.min_section, self.max_section = (
            self.section.unique().min(),
            self.section.unique().max(),
        )
        self.n_sections = int(self.max_section - self.min_section + 1)

        self.zz = (
            self.adata.obs["z_index"]
            if "z_index" in self.adata.obs.columns
            and not self.adata.obs["z_index"].isna().any()
            else self.adata.obs["y"]
        )
        self.yy = (
            -self.adata.obs["y_index"]
            if "y_index" in self.adata.obs.columns
            and not self.adata.obs["y_index"].isna().any()
            else -self.adata.obs["x"]
        )

    def KPCA_dimreduction(self):
        """
        Performs Kernel PCA to reduce dimensionality of the data and applies
        linear regression to find decoder weights.
        """

        mask = self.adata.varm[self.model.mask_key_]
        data = self.adata.X
        cov = np.cov(data.T)
        weight_matrix = compute_adjusting_weight_cov_matrix(cov, mask)
        spd_weight_matrix = pd.DataFrame(
            ensure_posdef(weight_matrix),
            index=self.adata.var_names,
            columns=self.adata.var_names,
        )

        adjusted_cov = pd.DataFrame(
            cov * spd_weight_matrix,
            index=self.adata.var.index,
            columns=self.adata.var.index,
        )

        n_components = len(self.adata.uns[self.model.mask_key_])

        # Non-Linear "Encoding"
        kpca = KernelPCA(n_components=n_components, kernel="precomputed")
        encoder_weights_kpca = kpca.fit_transform(adjusted_cov)
        encoder_weights_kpca = pd.DataFrame(
            encoder_weights_kpca, index=adjusted_cov.index
        )
        self.adata.obsm["KPCA_latent"] = np.dot(data, encoder_weights_kpca)

        # Approximate Decoder Weights
        lin_model = LinearRegression(fit_intercept=False)
        lin_model.fit(
            self.adata.obsm["KPCA_latent"], data
        )  # find A such that [X - (A * Y.T)]^2 is minimum

        self.adata.obsm["KPCA_output"] = np.dot(
            self.adata.obsm["KPCA_latent"], lin_model.coef_.T
        )

    def get_model_arch(self):
        """
        Displays the architecture of the model by listing the names and sizes of each parameter.
        """
        for name, p in self.model.model.named_parameters():
            print(name, " - ", p.size(0), p.size(-1))

    def plot_lba(self, color_key="lipotype_color", savepath=None):
        """
        Generates spatial plots of lipids colored by predefined categories.

        Parameters
        ----------
        color_key : str
            The key used to access the colors for each lipid in `adata.obs` (default is 'lipotype_color').
        savepath : str, optional
            Path to save the generated plot (default is None).
        """
        num_cols = 8
        num_rows = math.ceil(self.n_sections / num_cols)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows)
        )
        axes = axes.flatten()
        dot_size = 0.7

        for i in range(self.n_sections):
            ax = axes[i]
            ax.scatter(
                self.zz[self.section == i + 1],
                self.yy[self.section == i + 1],
                c=self.adata.obs[self.section == i + 1][color_key],
                s=dot_size,
                alpha=1,
                rasterized=True,
            )
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(self.zz.min(), self.zz.max())
            ax.set_ylim(self.yy.min(), self.yy.max())
            # ax.set_title(f"Section {i+1}", fontsize=20)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def plot_spatial(
        self,
        name,
        space,
        cmap="viridis",
        sym_colorscale=False,
        savepath=None,
    ):
        """
        Plots spatial distribution of a specified feature across different sections.

        Parameters
        ----------
        name : str
            Name of the lipid or feature to plot.
        space : str
            Specifies the type of data to plot ('input', 'output', 'residual', 'latent').
        cmap : str
            Colormap for the plot (default is 'viridis').
        sym_colorscale : bool
            Whether to use a symmetric color scale around zero (default is False).
        savepath : str, optional
            Path to save the generated plot (default is None).
        """

        assert space in ["input", "output", "residual", "latent"]
        assert (
            name in self.adata.uns[self.model.mask_key_]
            if space == "latent"
            else name in self.adata.var_names
        )

        index = (
            self.adata.uns[self.model.mask_key_].index(name)
            if space == "latent"
            else self.adata.var.index.get_loc(name)
        )
        to_plot = (
            self.adata.X[:, index]
            if space == "input"
            else self.adata.obsm[space][:, index]
        )
        q2, q98 = np.percentile(to_plot, [2, 98])

        num_cols = 8 if self.adata.obs["Sample"].nunique() == 1 else 6
        num_rows = math.ceil(self.n_sections / num_cols)

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(1 * num_cols, 1 * num_rows),
            rasterized=True,
        )
        axes = axes.flatten()
        dot_size = 0.7

        if sym_colorscale:
            vmin, vmax = -max(abs(q2), abs(q98)), max(abs(q2), abs(q98))
        else:
            vmin, vmax = q2, q98

        for s in range(self.n_sections):
            ax = axes[s]
            sc = ax.scatter(
                self.zz[self.section == self.min_section + s],
                self.yy[self.section == self.min_section + s],
                s=dot_size,
                alpha=1,
                c=to_plot[self.section == self.min_section + s],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(self.zz.min(), self.zz.max())
            ax.set_ylim(self.yy.min(), self.yy.max())
            # ax.set_title(f"Section {s+1}", fontsize=20)

        for j in range(s + 1, len(axes)):
            fig.delaxes(axes[j])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=q2, vmax=q98)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, cax=cbar_ax)

        fig.suptitle(f"{name} - {space}")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def plot_losses(self, savepath=None):
        """
        Plots normalized losses per lipid across training epochs,
        displaying both training and validation loss curves.

        Parameters
        ----------
        savepath : str, optional
            Path to save the generated plot (default is None).
        """
        # TODO: normalize the loss to obtain mean loss per lipid
        # --> otherwise we cannot compare reconstruction performances
        # between models with different numbers of input lipids
        assert (
            len(self.logs.keys()) % 2 == 0
        ), "The number of logs must be even, as they represent the training and validation losses"
        n_losses = int(len(self.logs.keys()) / 2)

        fig, axs = plt.subplots(n_losses, 1, figsize=(12, 10))
        loss_list = list(self.logs.keys())
        loss_types = [loss_list[i][6:] for i in range(n_losses)]

        for i, ax in enumerate(axs):
            ax.semilogy(
                np.sqrt(
                    np.array(self.logs[f"epoch_{loss_types[i]}"])
                    / self.model.input_dim_
                ),
                label=f"train_{loss_types[i]}",
            )
            ax.semilogy(
                np.sqrt(
                    np.array(self.logs[f"val_{loss_types[i]}"])
                    / self.model.input_dim_
                ),
                label=f"val_{loss_types[i]}",
            )
            ax.legend()
            ax.set_title(f"{loss_types[i]}", size=15)
            ax.set_ylabel("Loss")
            ax.set_xlabel("Epoch")

        # Show the plot
        plt.tight_layout()
        plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def plot_decoder_weights(self, binary=False, savepath=None):
        """
        Visualizes decoder weights as a heatmap, optionally binarizing the weights.

        Parameters
        ----------
        binary : bool
            If True, binarizes the decoder weights before plotting (default is False).
        savepath : str, optional
            Path to save the generated plot (default is None).
        """

        decoder_weights = (
            self.model.model.decoder.L0.expr_L.weight.data.cpu().numpy()
        )
        mask = self.model.model.mask.t()

        df_to_diagonalize = pd.DataFrame(
            decoder_weights if not binary else mask,
            columns=self.adata.uns[self.model.mask_key_],
            index=self.adata.var_names,
        )

        df_diagonalized, row_order, column_order = (
            diagonalize_heatmap_layout(df_to_diagonalize)
        )

        # Apply the same column order to mask
        mask = pd.DataFrame(
            mask[row_order, :][:, column_order],
            columns=[
                self.adata.uns[self.model.mask_key_][i]
                for i in column_order
            ],
            index=[self.adata.var_names[i] for i in row_order],
        )

        if not self.model.soft_mask_:
            df_diagonalized[mask == 0] = np.nan

        vmin = np.abs(np.nanmin(df_diagonalized.values))
        vmax = np.abs(np.nanmax(df_diagonalized.values))

        fig, ax1 = plt.subplots(figsize=(35, 35))
        sns.heatmap(
            df_diagonalized,
            cmap="PuRd" if self.model.soft_mask_ else "viridis",
            ax=ax1,
            xticklabels=df_diagonalized.columns,
            yticklabels=df_diagonalized.index,
            vmin=vmin,
            vmax=vmax,
            mask=np.isnan(df_diagonalized.values),
        )

        # Customize ticks
        ax1.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
        )
        ax1.tick_params(
            axis="y", which="both", left=False, right=False, pad=20
        )
        ax1.set_title(
            (
                f"Decoder weights - Mask Membership: SOFT - ALPHA L1: {self.model.training_params['alpha_l1']}"  # self.model.alpha_l1
                if self.model.soft_mask_
                else "Decoder weights - Mask Membership: HARD"
            ),
            fontsize=30,
        )
        plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def to_pdf(
        self, space, savepath, cmap="viridis", sym_colorscale=False
    ):
        """
        Exports a series of spatial plots to a PDF file, each page represents a different feature.

        Parameters
        ----------
        space : str
            The type of data space to plot ('input', 'output', 'residual', 'latent').
        savepath : str
            File path to save the PDF document.
        cmap : str
            Colormap to use for plotting (default is 'viridis').
        sym_colorscale : bool
            If True, uses a symmetric color scale (default is False).
        """
        assert (
            space
            in [
                "input",
                "output",
                "residual",
                "latent",
            ]
        ), "space must be either 'input' or 'output' or 'residual' or 'latent'"

        pdf_pages = PdfPages(savepath)

        elements = (
            self.adata.uns[self.model.mask_key_]
            if space == "latent"
            else self.adata.var_names
        )

        for i, curr in tqdm(enumerate(elements)):
            results = []
            filtered_data = pd.concat(
                [
                    pd.concat([self.zz, self.yy, self.section], axis=1),
                    pd.DataFrame(
                        self.adata.X[:i]
                        if space == "input"
                        else self.adata.obsm[space][:, i],
                        index=self.adata.obs_names,
                        columns=[curr],
                    ),
                ],
                axis=1,
            )

            for s in range(self.n_sections):
                subset = filtered_data[
                    filtered_data["SectionID"] == self.min_section + s
                ]
                perc_2 = subset[curr].quantile(0.02)
                perc_98 = subset[curr].quantile(0.98)
                results.append([s + 1, perc_2, perc_98])

            percentile_df = pd.DataFrame(
                results, columns=["Section", "2-perc", "98-perc"]
            )
            med2p = percentile_df["2-perc"].median()
            med98p = percentile_df["98-perc"].median()

            if sym_colorscale:
                vmin, vmax = (
                    -max(abs(med2p), abs(med98p)),
                    max(abs(med2p), abs(med98p)),
                )
            else:
                vmin, vmax = med2p, med98p

            # Create the subplots
            num_cols = 8 if self.adata.obs["Sample"].nunique() == 1 else 6
            num_rows = math.ceil(self.n_sections / num_cols)

            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(3 * num_cols, 3 * num_rows),
                rasterized=True,
            )

            axes = axes.flatten()
            for s in range(self.n_sections):
                ax = axes[s]
                ddf = filtered_data[
                    (filtered_data["SectionID"] == self.min_section + s)
                ]
                ax.scatter(
                    self.zz[
                        self.section == self.min_section + s
                    ],  # 'zccf'
                    self.yy[self.section == self.min_section + s],  # 'yccf
                    c=ddf[curr],
                    cmap=cmap,
                    s=0.5,
                    rasterized=True,
                    vmin=vmin,
                    vmax=vmax,
                )

            for ax in axes:
                ax.axis("off")
                ax.set_aspect("equal")

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, cax=cbar_ax)

            fig.suptitle(curr)
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            pdf_pages.savefig(fig)
            plt.close(fig)

        pdf_pages.close()

    def plot_LPs_vs_feature(self, feature="region", savepath=None):
        """
        Creates a heatmap displaying the variation of lipid programs
        across a specific categorical feature.

        Parameters
        ----------
        feature : str
            The observation feature to analyze, typically a categorical variable such as 'region'.
        savepath : str, optional
            Path to save the generated plot (default is None).
        """

        mean_latent_coord = (
            pd.DataFrame(
                self.adata.obsm["latent"],
                index=self.adata.obs_names,
                columns=self.adata.uns[self.model.mask_key_],
            )
            .groupby(self.adata.obs[feature])
            .mean()
        )

        mean_latent_coord_cent = (
            mean_latent_coord - mean_latent_coord.mean(axis=0)
        )

        a = np.percentile(mean_latent_coord_cent.values, 5)
        b = np.percentile(mean_latent_coord_cent.values, 95)

        extr = np.max([np.abs(a), np.abs(b)])
        mean_latent_coord_cent = mean_latent_coord_cent.clip(
            lower=a, upper=b
        )

        mean_latent_coord_cent, _, _ = diagonalize_heatmap_layout(
            mean_latent_coord_cent
        )

        mean_latent_coord_cent.columns = (
            mean_latent_coord_cent.columns.astype(str)
        )
        mean_latent_coord_cent.index = mean_latent_coord_cent.index.astype(
            str
        )

        fig, ax1 = plt.subplots(figsize=(25, 20))
        sns.heatmap(
            mean_latent_coord_cent,
            cmap="PiYG",
            ax=ax1,
            xticklabels=[
                col.replace("_", " ")
                for col in mean_latent_coord_cent.columns
            ],
            yticklabels=[
                idx.replace("_", " ")
                for idx in mean_latent_coord_cent.index
            ],
            vmin=-extr,
            vmax=extr,
        )
        ax1.tick_params(axis="x", which="both", bottom=False, top=False)
        ax1.tick_params(
            axis="y", which="both", left=False, right=False, pad=20
        )

        ax1.set_xlabel("Lipid Programs", fontsize=15)
        ax1.set_ylabel(f"{feature}", fontsize=15)

        # Set title
        ax1.set_title(
            f"Hyper and Hypo Activity of LPs {feature}-wise", fontsize=20
        )

        plt.show()

        if savepath is not None:
            plt.savefig(savepath)

    def plot_tsne():
        # TODO: implement (see discussion_thesis.ipynb and prove.ipynb)
        pass

    def plot_umap():
        # TODO: implement (see final_pipeline.ipynb and prove.ipynb)
        pass

    def plot_latent_correlation():
        # TODO: implement (see provaprova.ipynb)
        pass
