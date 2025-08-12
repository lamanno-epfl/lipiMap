import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

import os

from tqdm import tqdm


def merge_lmt_files(masks, masks_path):
    """
    Merges multiple lipid matrix files (.lmt) into a single combined file named 'all.lmt'.

    Parameters
    ----------
    masks : list of str or str
        A list of block identifiers corresponding to the masks in the `.lmt` files.
        If a single string is provided, it will be converted into a list.
    masks_path : str
        Path to the directory containing the `.lmt` files.

    Returns
    -------
    tuple
        A tuple containing:
        - List of file paths, including the original `.lmt` files plus the newly created 'all.lmt'.
        - List of block identifiers, modified to include 'all' as the last element.
    """

    masks = [masks] if isinstance(masks, str) else masks
    masks.append("all")
    files = [f"{masks_path}/{m}.lmt" for m in masks]

    with open(files[-1], "w") as all_file:  # 'all.lmt'
        for f in files[:-1]:
            with open(f) as infile:
                for line in infile:
                    all_file.write(line)

    return files, masks


def add_annotations(adata, masks_path, masks="all", min_lipids=2):
    """
    Adds lipid program annotations to an AnnData object based on `.lmt` files.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. The lipids should be in `adata.var_names`.
    masks_path : str
        Path to the directory containing `.lmt` files.
    masks : str or list of str, optional (default: 'all')
        Block identifiers corresponding to `.lmt` files to include.
        If 'all', the function merges all `.lmt` files into a single file.
    min_lipids : int, optional (default: 2)
        Minimum number of lipids required for a program to be included.

    Returns
    -------
    None
        Updates the AnnData object `adata` in place with:
        - `adata.varm[masks[i]]`: Binary membership matrix (n_vars x n_programs),
          where `I[i, j] = 1` if lipid `i` is part of program `j`.
        - `adata.uns[masks[i]]`: List of program names corresponding to the columns of the membership matrix.

    Notes
    -----
    - Each `.lmt` file is expected to contain lipid programs, where the first column is the program name,
      followed by lipid names in the same row.
    - Programs that contain fewer than `min_lipids` lipids in the `adata` object are excluded.
    - The function automatically merges multiple `.lmt` files if required and stores the combined results
      under the block identifier 'all' in the AnnData object.
    """
    files, masks = merge_lmt_files(masks, masks_path)
    annot = []

    for i, file in enumerate(files):
        with open(file) as f:
            programs = [l.strip("\n").split("\t") for l in f]
        annot.append(programs)

        I = np.asarray(
            [
                [int(lipid in prog) for prog in programs]
                for lipid in adata.var_names
            ],
            dtype="int32",
        )

        # Only select the programs that have at least 'min_lipids' lipids
        mask = I.sum(0) >= min_lipids  # [i]

        I = I[:, mask]  # --> n_vars x n_actual_progs

        adata.varm[masks[i]] = I
        adata.uns[masks[i]] = [
            prog[0]
            for i, prog in enumerate(programs)
            if i not in np.where(~mask)[0]
        ]


def add_lipid_program_enrichment(adata, mask_key):
    """
    Computes and adds lipid program enrichment scores to an AnnData object based on standardized lipid expression data.
    The formula for calculating the lipid program enrichment for a given program is:

        lp_enrichment = mean((X_lp - mean(X_lp)) / std(X_lp))

    where `X_lp` represents the expression values of lipids associated with the lipid program, `mean(X_lp)` is the mean expression
    value of those lipids, and `std(X_lp)` is the standard deviation of those expression values.

    Parameters:
    ----------
    adata : AnnData
        An AnnData object that contains, at minimum, a `.X` attribute with lipid expression data, a `.varm` attribute containing
        binary masks for lipid programs, and a `.uns` attribute with names of the lipid programs.
    mask_key : str
        A key in `adata.varm` and `adata.uns` that identifies the binary mask and the corresponding lipid program names.

    Returns:
    -------
    AnnData
        The modified AnnData object with added lipid program enrichment columns in `adata.obs`.
    """

    mask = adata.varm[mask_key]
    lipid_programs_names = adata.uns[mask_key]
    lipid_indices = [
        np.where(mask[:, col] == 1)[0] for col in range(mask.shape[1])
    ]
    means, stds = adata.X.mean(axis=0), adata.X.std(axis=0)

    for j, lp in enumerate(lipid_programs_names):
        lipids_idx = lipid_indices[j]
        lipids_standardized = (
            adata.X[:, lipids_idx] - means[lipids_idx]
        ) / stds[lipids_idx]
        # lp_enrichment = lipids_standardized.sum(axis=1)
        lp_enrichment = lipids_standardized.mean(axis=1)
        adata.obs[f"{lp}_enrichment"] = lp_enrichment

    return adata


def remove_latent_collinearity(adata, mask_key):
    """
    Removes collinear lipid programs from the AnnData object based on Hamming distances in the binary mask.
    It calculates the Hamming distance between all pairs of lipid programs to identify those that are identical
    (zero distance). It then merges identical lipid programs into a single program and updates the AnnData object accordingly.

    Parameters:
    ----------
    adata : AnnData
        An AnnData object containing the following:
        - `adata.varm[mask_key]`: a binary mask where columns correspond to lipid programs and rows to lipids.
        - `adata.uns[mask_key]`: a list of names corresponding to the lipid programs.
    mask_key : str
        The key used to access lipid program information in `adata.varm` and `adata.uns`.

    Returns:
    -------
    None
        Modifies the AnnData object in place by updating `adata.varm` and `adata.uns` to remove collinear lipid programs.
    """
    mask = adata.varm[mask_key].copy()
    hamming_distances = pdist(mask.T, metric="hamming")
    zero_indices = np.where(hamming_distances == 0)[0]
    hamming_matrix = squareform(hamming_distances)
    upper_tri_indices = np.triu_indices(hamming_matrix.shape[0], k=1)
    identical_programs = find_duplicates(
        upper_tri_indices[0][zero_indices],
        upper_tri_indices[1][zero_indices],
    )
    to_remove = []
    for group in identical_programs:
        adata.uns[mask_key][list(group)[0]] = " + ".join(
            [adata.uns[mask_key][i] for i in group]
        )
        to_remove.append(list(group)[1:])
    adata.varm[mask_key] = np.delete(
        mask, np.concatenate(to_remove), axis=1
    )
    adata.uns[mask_key] = [
        name
        for i, name in enumerate(adata.uns[mask_key])
        if i not in np.concatenate(to_remove)
    ]


def find_duplicates(first_array, second_array):
    G = nx.Graph()
    for a, b in zip(first_array, second_array):
        G.add_edge(a, b)
    return list(nx.connected_components(G))


def _get_lp_representation_score(adata, lp, count, sep_lp, verbose=True):
    score = (count * sep_lp.shape[0]) / sep_lp["BrainLipids"].sum()
    if np.isnan(score):
        assert (
            lp in adata.uns["LBA_lipid_families"]
        ), "Lipid family not found in lipid programs"
        if verbose:
            print(
                f"{lp} is not in LION, representation score set to 0 by default"
            )  # noqa: E701
        score = 0
    return score


def compute_representation_score(
    adata, lion_lipid_programs, multi_index_data, mask_key
):
    """
    Computes the lipid pr-->{color}
    This function calculates an absolute each li-->{color} of matching counts in the data and the cardinality of lipid programs in LION database.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing lipid data. The `mask_key` in `adata.varm` stores a matrix
        where rows correspond to lipid features and columns correspond to lipid programs.
        The `uns[mask_key]` stores lipid program names or combinations.

    lipid_programs : pandas.DataFrame
        A DataFrame containing lipid program information. Must include columns:
        - 'LION_Name': Names of lipid programs.
        - 'LION_Code': Corresponding codes of lipid programs.

    multi_index_data : pandas.DataFrame
        A DataFrame with a multi-index column structure. The column level 'Level 4' should contain
        the lipid program codes used to compute cardinality.

    mask_key : str
        The key for accessing lipid program assignments in `adata.varm` and `adata.uns`.

    Returns
    -------
    None
        The function adds the computed representation scores to `adata.uns` with the key `'representation_score'`.
    """

    # Extract the set of lipid programs annotated in the input dataset
    # Consider as separate programs the ones that are combined with ' + '
    set_lps = {
        lp for mask in adata.uns[mask_key] for lp in mask.split(" + ")
    }  # 60 in total (starting from 49)

    lion_lps = lion_lipid_programs[
        lion_lipid_programs["LION_Name"].isin(set_lps)
    ]  # noqa: E501
    brainlip = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "../data/LION_Data/brainlip.csv"
        )
    )

    lipids_codes = [
        multi_index_data.loc[
            :, multi_index_data.columns.get_level_values("Level 4") == col
        ][
            multi_index_data.loc[
                :,
                multi_index_data.columns.get_level_values("Level 4")
                == col,
            ]
            == 1
        ]
        .dropna()
        .index
        for col in lion_lps["LION_Code"].values
    ]
    lion_lps["Lipids"] = lipids_codes
    lion_lps["BrainLipids"] = np.array(
        [
            sum(
                [
                    code in set(brainlip["LION_Code"].dropna().values)
                    for code in codes
                ]
            )
            for codes in lipids_codes
        ]
    )

    representation_scores = []
    counts = adata.varm[mask_key].sum(0)
    representation_scores = [
        _get_lp_representation_score(
            adata,
            lp,
            counts[i],
            lion_lps[lion_lps["LION_Name"].isin(lp.split(" + "))],
        )
        for i, lp in enumerate(adata.uns[mask_key])
    ]

    adata.uns["representation_score"] = np.array(representation_scores)


def compute_exclusivity_score(adata, mask_key):
    """
    Computes the exclusivity score for lipid programs. The exclusivity score measures how
    uniquely a lipid program is assigned to individual lipids in the AnnData object.
    Higher scores indicate greater exclusivity.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing lipid program data. The `mask_key` in `adata.varm` should
        store a matrix where rows correspond to lipid features and columns to lipid programs.

    mask_key : str
        The key for accessing lipid program assignments in `adata.varm`.

    Returns
    -------
    None
        The exclusivity scores are stored in `adata.uns['exclusivity_score']` as a list.
    """

    mask_scores = adata.varm[mask_key].copy().astype(np.float64)
    mask_scores *= (1 - mask_scores.sum(axis=1) / mask_scores.shape[1])[
        :, np.newaxis
    ]
    adata.uns["exclusivity_score"] = mask_scores.max(axis=0)

    # alternative
    # mask_scores = adata.varm[mask_key].copy().astype(np.float64) / adata.varm[mask_key].sum(axis=1)
    # adata.uns['exclusivity_score'] = mask_scores.mean(axis=0)


def compute_density_score(adata, mask_key, opt_density=0.4, tolerance=0.2):
    """
    Computes the density score for lipid programs. The density score evaluates how close
    the lipid program's density is to the optimal density, with a Gaussian penalty for deviations.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing lipid program data. The `mask_key` in `adata.varm` should
        store a matrix where rows correspond to lipid features and columns to lipid programs.

    mask_key : str
        The key for accessing lipid program assignments in `adata.varm`.

    opt_density : float, optional
        The optimal density value for lipid programs. Default is 0.4.

    tolerance : float, optional
        The standard deviation of the Gaussian penalty. Default is 0.2.

    Returns
    -------
    None
        The density scores are stored in `adata.uns['density_score']` as a list.
    """

    lp_density = (
        adata.varm[mask_key].sum(axis=0) / adata.varm[mask_key].shape[0]
    )
    adata.uns["density_score"] = np.exp(
        -((lp_density - opt_density) ** 2) / (2 * tolerance**2)
    )


def rank_LPs(adata, weights=[0.75, 0.15, 0.1]):
    """
    Ranks lipid programs based on their representation, exclusivity, and density scores.
    The function computes the final score for each lipid program by combining the representation,
    exclusivity, and density scores with the provided weights.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing lipid program data. The `uns` attribute should contain
        representation, exclusivity, and density scores.

    weights : list of float, optional
        A list of three floats representing the weights for representation, exclusivity, and density scores, respectively.
        Default is [0.75, 0.15, 0.1].

    Returns
    -------
    None
        The function adds a 'final_score' column to the `adata.uns` attribute with the final score for each lipid program.
    """

    adata.uns["final_score"] = (
        weights[0] * adata.uns["representation_score"]
        + weights[1] * adata.uns["exclusivity_score"]
        + weights[2] * adata.uns["density_score"]
    )
    # alternative
    # adata.uns['final_score'] = adata.uns['representation_score'] * adata.uns['exclusivity_score'] * adata.uns['density_score']

    # check if the sorting of lps is the same with the 2 different computations
    # print("weighted_sum", np.argsort(adata.uns["final_score"]))
    # print(
    #     "multiplication",
    #     np.argsort(
    #         adata.uns["representation_score"]
    #         * adata.uns["exclusivity_score"]
    #         * adata.uns["density_score"]
    #     ),
    # )


def representation_analysis(
    adata,
    LION_data_handler,
    multi_index_data,
    mask_key,
    n_replicas=10000,
    seed=0,
):
    """
    Simulates bootstrap replicas of lipids to estimate the representativeness of lipid programs within the dataset.
    The function generates a distribution of lipid program representation scores across the replicas.

    Parameters:
    -----------
    adata : anndata.AnnData
        The annotated data matrix (with var_names as lipids) for which lipid programs' distribution is to be analyzed.
    LION_data_handler : object
        An object that handles operations on lipid data, including filtering and aggregation based on a specified refinement level.
    multi_index_data : pandas.DataFrame
        A multi-index dataframe containing lipid data, where the index represents lipids and columns contain metadata such as lipid programs.
    mask_key : str
        A key from `adata.uns` that contains the names of lipid programs to be analyzed.

    Returns:
    --------
    dict
        A dictionary where each key is a lipid program name, and the value is a list of proportions representing the lipid program's
        presence in each subsample across the iterations.
    """
    lion_lipid_programs = LION_data_handler.lipid_programs
    n = len(adata.var_names)
    set_lps = {
        lp for mask in adata.uns[mask_key] for lp in mask.split(" + ")
    }  # 60 in total (starting from 49)

    lion_lps = lion_lipid_programs[
        lion_lipid_programs["LION_Name"].isin(set_lps)
    ]  # noqa: E501
    brainlip = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), "../data/LION_Data/brainlip.csv"
        )
    )

    lipids_codes = [
        multi_index_data.loc[
            :, multi_index_data.columns.get_level_values("Level 4") == col
        ][
            multi_index_data.loc[
                :,
                multi_index_data.columns.get_level_values("Level 4")
                == col,
            ]
            == 1
        ]
        .dropna()
        .index
        for col in lion_lps["LION_Code"].values
    ]
    lion_lps["Lipids"] = lipids_codes
    lion_lps["BrainLipids"] = np.array(
        [
            sum(
                [
                    code in set(brainlip["LION_Code"].dropna().values)
                    for code in codes
                ]
            )
            for codes in lipids_codes
        ]
    )

    distribution = {lp_name: [] for lp_name in adata.uns[mask_key]}

    np.random.seed(seed)

    print("Generating replicas...")
    for r in tqdm(range(n_replicas)):
        # Draw the subsample whose cardinality is the same of the input data annotation (i.e. number of lipids)
        subsample = np.random.choice(
            brainlip["LION_Code"].dropna(), n, replace=False
        )

        for i, lp in enumerate(adata.uns[mask_key]):
            sep_lp = lion_lps[lion_lps["LION_Name"].isin(lp.split(" + "))]
            count = (
                sep_lp["Lipids"]
                .apply(lambda x: sum([code in subsample for code in x]))
                .mean()
            )
            distribution[lp].append(
                _get_lp_representation_score(
                    adata, lp, count, sep_lp, verbose=False
                )
            )

    return distribution


def representation_filtering(
    adata,
    mask_framework,
    distribution,
    lb=40,
    ub=50,
    plot=True,
):
    """
    Generates a violin plot showing the distribution of lipid programs and their representation scores.

    Parameters:
    -----------
    adata : anndata.AnnData
        The annotated data matrix containing uns with keys for 'final_score' and 'representation_score' and the lipid programs under mask_framework.
    distribution : dict
        A dictionary where keys are lipid program names and values are lists of representation values.
    mask_framework : str
        A key in `adata.uns` to access the list of lipid program names for which to plot the distribution.
    plot : bool, optional
        If True, the function generates a violin plot. Default is True.
    Returns:
    --------
    Displays a violin plot.
    """

    sorted_indices = np.argsort(adata.uns["final_score"])

    lower = [
        np.percentile(distribution[lp], lb) for lp in distribution.keys()
    ]
    upper = [
        np.percentile(distribution[lp], ub) for lp in distribution.keys()
    ]

    # Plot all the lp values with horizontal violin plots
    positions = np.arange(len(adata.uns[mask_framework]))

    to_keep = []

    print("Filtering lipid programs based on representation score...")
    for pos, i in zip(positions, sorted_indices):
        lp = adata.uns[mask_framework][i]
        if lp not in distribution:
            distribution[lp] = np.zeros(
                distribution[next(iter(distribution))].shape
            )
        # Determine the color based on the representation_score
        score = adata.uns["representation_score"][i]
        if score > upper[i]:
            color = "green"
        elif lower[i] <= score <= upper[i]:
            color = "orange"
        else:
            color = "red"

        if lp in adata.uns["LBA_lipid_families"]:
            color = "black"

        if color == "green" or color == "black":
            to_keep.append(lp)

    print(f"Number of lipid programs after filtering: {len(to_keep)}")

    if plot:
        print("\nPlotting representation scores and distributions ...")
        fig, ax = plt.subplots(figsize=(15, 25))
        for pos, i in zip(positions, sorted_indices):
            # Set the color for the violins and the median/extreme lines
            lp = adata.uns[mask_framework][i]

            score = adata.uns["representation_score"][i]
            if score > upper[i]:
                color = "green"
            elif lower[i] <= score <= upper[i]:
                color = "orange"
            else:
                color = "red"

            if lp in adata.uns["LBA_lipid_families"]:
                color = "black"

            parts = ax.violinplot(
                distribution[lp],
                positions=[pos],
                vert=False,
                widths=0.6,
                showmeans=False,
                showmedians=True,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
            parts["cmedians"].set_edgecolor(color)
            parts["cmedians"].set_linewidth(2)
            parts["cbars"].set_edgecolor(color)
            parts["cbars"].set_linewidth(2)
            parts["cmins"].set_edgecolor(color)
            parts["cmins"].set_linewidth(2)
            parts["cmaxes"].set_edgecolor(color)
            parts["cmaxes"].set_linewidth(2)

            # Plot the actual representation score
            ax.scatter(
                score,
                pos,
                color="black",
                zorder=3,
                label="Representation Score" if pos == 0 else "",
            )

        # Set the y-axis labels to be horizontal
        ax.set_yticks(positions)
        ax.set_yticklabels(
            [adata.uns[mask_framework][i] for i in sorted_indices],
            fontsize=10,
        )

        # Add labels and title
        ax.set_xlabel("Values")
        ax.set_ylabel("Lipid Programs")
        ax.set_title("LBA vs LION - Representation Analysis")

        # Create custom legend handles
        green_patch = mpatches.Patch(
            color="green", label=f"LBA > {ub}th percentile"
        )
        orange_patch = mpatches.Patch(
            color="orange", label=f"{lb}th <= LBA <= {ub}th percentile"
        )
        red_patch = mpatches.Patch(
            color="red", label=f"LBA < {lb}th percentile"
        )
        black_patch = mpatches.Patch(
            color="black", label="Representation Score"
        )

        # Add legend
        ax.legend(
            handles=[green_patch, orange_patch, red_patch, black_patch],
            loc="upper right",
        )

        plt.tight_layout()
        plt.show()

    return to_keep


# TODO: Implement the following functions
# def perform_ORA(adata, LION_data_handler, mask_framework, files):
#     df = pd.read_csv(LION_data_handler.matching_path)
#     lipids_of_interest = set(df[df['ProjectLipid'].isin(adata.var_names)]['ProjectLipid'])

#     reference_lipids = set(LION_data_handler.leaf_level_0_paths_dict.keys())

#     with open(LION_data_handler.lipids_path, 'r') as f:
#         lines = f.readlines()

#     lipid_mapping = pd.DataFrame([(line.split('\t')[0], line.split('\t')[1].strip()) for line in lines], columns=['lipid_name', 'lipid_code'])
#     reference_lipids_names = set(lipid_mapping[lipid_mapping['lipid_code'].isin(reference_lipids)]['lipid_name'])

#     lion_annotations = defaultdict(set)

#     lion_lmt = [s for s in files if 'lipidontology' in s][0]
#     with open(lion_lmt, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             lipid_program = parts[0]
#             lipids = parts[1:]
#             for lipid in lipids:
#                 lion_annotations[lipid].add(lipid_program)

#     reference_LION_counts = defaultdict(int)
#     for lipid, terms in lion_annotations.items():
#         if lipid in reference_lipids_names:
#             for term in terms:
#                 reference_LION_counts[term] += 1

#     lion_annotations_set = set(lion_annotations.keys())
#     outside_intersection = reference_lipids_names.symmetric_difference(lion_annotations_set)

#     interest_LION_counts = defaultdict(int)
#     for lipid, terms in lion_annotations.items():
#         if lipid in lipids_of_interest or lipid.replace('_', '(') + ')' in lipids_of_interest or lipid in outside_intersection:
#             for term in terms:
#                 interest_LION_counts[term] += 1

#     # Perform Fisher's Exact Test
#     results = []
#     for term, interest_count in interest_LION_counts.items():
#         reference_count = reference_LION_counts.get(term, 0)
#         table = [
#             [interest_count, len(lipids_of_interest) - interest_count],
#             [reference_count, len(reference_lipids) - reference_count]
#         ]
#         odds_ratio, p_value = fisher_exact(table, alternative='greater')
#         results.append((term, interest_count, reference_count, p_value))

#     results_df = pd.DataFrame(results, columns=["LION Term", "Interest Count", "Reference Count", "p-value"])
#     results_df['adj_p-value'] = multipletests(results_df['p-value'], method='fdr_bh')[1]

#     adata.uns['ora_values'] = [[] for _ in range(len(adata.uns[mask_framework]))]

#     for lp_lion in results_df['LION Term']:
#         try:
#             lp = min([s for s in adata.uns[mask_framework] if lp_lion in s], key=len)
#             adata.uns['ora_values'][adata.uns[mask_framework].index(lp)].append(results_df[results_df['LION Term']==lp_lion]['adj_p-value'].values[0])
#         except:
#             continue

#     adata.uns['ora_values'] = [statistics.mean(elem) if elem else 0 for elem in adata.uns['ora_values']]


def manual_programs_removal(adata, mask_framework, to_remove):
    to_remove = set(adata.uns[mask_framework]).intersection(to_remove)
    indices_to_remove = [
        adata.uns[mask_framework].index(prog) for prog in to_remove
    ]
    adata.varm[mask_framework] = np.delete(
        adata.varm[mask_framework], indices_to_remove, axis=1
    )
    adata.uns[mask_framework] = [
        adata.uns[mask_framework][i]
        for i in range(len(adata.uns[mask_framework]))
        if i not in indices_to_remove
    ]


def manual_sections_removal(adata, to_remove):
    for i in to_remove:
        adata = adata[adata.obs["Section"] != i]
