# Imports
import os
import re

from collections import defaultdict
import anndata
import numpy as np


class LBADataHandler:
    """
    A handler for preprocessing and managing MALDI-MSI lipidomics data.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing lipid expression data and metadata.
    masks_path : str, optional
        Path to the directory containing mask data (default is './masks').
    initial_format : {'log', 'exp', 'norm_exp'}, optional
        The format of the input lipid data (default is 'log').
    final_format : {'log', 'exp', 'norm_exp'}, optional
        The desired format for the lipid data after transformation (default is 'norm_exp').
    amplify : bool, optional
        Whether to amplify the normalized data to mimic raw counts (default is True).
    """

    def __config__(self, masks_path, df):
        """
        Configures paths for processing data.

        Parameters
        ----------
        masks_path : str
            Path to the directory containing mask data.
        """

        print("Configuring Data Processor...\n")
        self.masks_path = masks_path
        self.lmt_path = os.path.join(self.masks_path, "LBA")

        self.df = df.copy()
        print(f"Data Shape: {self.df.shape}\n")

    def __init__(
        self,
        df,
        masks_path="./masks",
        initial_format="log",
        final_format="norm_exp",
        amplify=True,
    ):
        """
        Initializes the DataHandler class and preprocesses the input data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with lipid expression data.
        masks_path : str, optional
            Path to the directory containing mask data (default is './masks').
        initial_format : {'log', 'exp', 'norm_exp'}, optional
            The format of the input lipid data (default is 'log').
        final_format : {'log', 'exp', 'norm_exp'}, optional
            Desired format for lipid data after transformation (default is 'norm_exp').
        amplify : bool, optional
            Whether to amplify normalized data to mimic raw counts (default is True).
        """

        self.__config__(masks_path, df)

        self._extract_lipid_families()

        self.lipid_columns = []
        self.other_columns = []
        self.__fill_lipid_meta_cols()

        self._transform_data(initial_format, final_format)
        self._raw_counts_like(final_format, amplify)

        self.nn, self.count, self.proportion = self.nn_count_proportion()

    def _extract_lipid_families(self):
        """
        Extracts unique lipid family names from the dataframe columns.
        """

        self.lipid_families = set()
        pattern = r"^([A-Z][a-zA-Z]*)\s"

        for col in self.df.columns:
            match = re.match(pattern, col)
            if match:
                self.lipid_families.add(match.group(1))

        print(f"Lipid Families: {self.lipid_families}\n")

    def __fill_lipid_meta_cols(self):
        """
        Separates lipid expression columns and metadata columns.
        """

        self.lipid_columns = [
            col
            for col in self.df.columns
            if any([l_fam in col for l_fam in self.lipid_families])
        ]
        print(f"{len(self.lipid_columns)} Lipid Expressions Columns")

        self.other_columns = [
            col
            for col in self.df.columns
            if all([l_fam not in col for l_fam in self.lipid_families])
        ]
        print(f"{len(self.other_columns)} Other Columns (Metadata)\n")

    def _transform_data(self, initial_format, final_format):
        """
        Transforms the lipid data between log, exp, and normalized formats.

        Parameters
        ----------
        initial_format : {'log', 'exp', 'norm_exp'}
            The format of the input data.
        final_format : {'log', 'exp', 'norm_exp'}
            The desired format of the data.

        Raises
        ------
        ValueError
            If `initial_format` or `final_format` is invalid.
        """
        if initial_format == "log" and final_format == "exp":
            self.df[self.lipid_columns] = np.exp(self.df[self.lipid_columns])

        elif initial_format == "exp" and final_format == "log":
            self.df = np.log(self.df)

        elif initial_format == "log" and final_format == "norm_exp":
            for col in self.lipid_columns:
                self.df[col] = np.exp(self.df[col])
                self.df[col] = (self.df[col] - self.df[col].min()) / (
                    self.df[col].max() - self.df[col].min()
                )

        elif initial_format == "exp" and final_format == "norm_exp":
            for col in self.lipid_columns:
                self.df[col] = (self.df[col] - self.df[col].min()) / (
                    self.df[col].max() - self.df[col].min()
                )
        else:
            raise ValueError(
                "initial_format and final_format must be 'log', 'exp' or 'norm_exp'"
            )

        print(f"Data transformed from {initial_format} to {final_format}")

    def nn_count_proportion(self):
        """
        Identifies NN, count, and proportion columns in the dataframe.

        Returns
        -------
        tuple
            A tuple containing lists of columns:
            (nn_columns, count_columns, proportion_columns).
        """
        nn = [col for col in self.df.columns if col.startswith("NN")]
        count = [col for col in self.df.columns if col.startswith("count_")]
        proportion = [col for col in self.df.columns if col.startswith("proportion_")]
        return nn, count, proportion

    def _raw_counts_like(self, final_format, amplify):
        """
        Mimics raw count data by amplifying normalized data.

        Parameters
        ----------
        final_format : {'norm_exp'}
            The final format of the data.
        amplify : bool
            Whether to amplify the data.
        """
        if final_format == "norm_exp" and amplify:
            self.df[self.lipid_columns] = (
                (self.df[self.lipid_columns] * 1000).round().astype(int)
            )
            print("Data amplified by 1000 to mimic gene expression data (raw counts)")

    def to_anndata(self, final_format="norm_exp"):
        """
        Converts the dataframe into an AnnData object.

        Parameters
        ----------
        final_format : {'log', 'exp', 'norm_exp'}, optional
            Desired format for the lipid data (default is 'norm_exp').

        Returns
        -------
        anndata.AnnData
            An AnnData object containing the lipid data and metadata.
        """
        assert final_format in [
            "log",
            "exp",
            "norm_exp",
        ], "final_format must be 'log', 'exp' or 'norm_exp'"

        cols = self.lipid_columns.copy()
        X_data = self.df[cols].astype("float32")

        # Select the columns for .obs attribute
        obs_data = self.df.drop(columns=cols)

        # Create an AnnData object
        adata = anndata.AnnData(X=X_data, obs=obs_data)

        return adata

    def create_family_lmt(self):
        """
        Creates a `.lmt` (Lipid Matrix Transposed) file that maps lipid families 
        to their associated lipids.

        This method organizes lipids into families based on the first word in 
        their names (assumed to be the family identifier) and writes the mappings 
        to a tab-delimited Lipid Matrix Transposed (`.lmt`) file.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Writes a `.lmt` file named `<self.lmt_path>_lipid_families.lmt` to the specified path.

        File Format
        ------------
        The output `.lmt` file contains one row per lipid family. Each row lists the family name,
        followed by the associated lipids. The file format is tab-delimited. Example:
        ```
        Family_1    Lipid_1    Lipid_2    Lipid_3
        Family_2    Lipid_4    Lipid_5
        ```
        """
        families = defaultdict(set)
        for lipid in self.lipid_columns:
            family = lipid.split(" ")[0]
            families[family].add(lipid)

        with open(self.lmt_path + "_lipid_families.lmt", "w") as f:
            for family, lipids in families.items():
                f.write(f"{family}\t" + "\t".join(lipids) + "\n")
