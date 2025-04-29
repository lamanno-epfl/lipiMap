# Imports
import json
import os
import pickle
from itertools import accumulate
from tqdm import tqdm

import numpy as np
import pandas as pd

### LION Class
# TODO input data: check which of them can be dowloaded from the LION web page !!!
# TODO CHECK: I think LION.csv can be dowloaded from the LION web page, maybe also LION_association.txt, but not LION_Lipids.txt and LipidProgramsHierarchy.csv !!!
# - TODO LION_association.txt (https://github.com/martijnmolenaar/LION-web/blob/master/OntologyApp/data/20190704%20LION_association.txt)... better LION_Lipids.txt or directly the LION.csv (we can extract the lipid name and the lion codes from LION.csv)
# - TODO LipidProgramsHierarchy.csv (https://github.com/martijnmolenaar/LION-web/blob/master/OntologyApp/data/20191010%20LION_tree_structure.R) (another file can be dowloaded, but this version in very needed for the code), try to describe how to manually obtain this one from the one you can dowload
# - LION.csv (https://bioportal.bioontology.org/ontologies/LION/?p=summary)
# - TODO multi_index_list.pkl (this can be obtained from LipidProgramsHierarchy.csv, maybe come up with a snippet of code to do this)


class LIONDataHandler:
    """
    A handler for processing and managing hierarchical lipidomics data using the LION database.

    Parameters
    ----------
    save_data_path : str, optional
        Path to the directory for saving processed data (default is './data').
    masks_path : str, optional
        Path to the directory containing masks (default is './masks').
    save_data : bool, optional
        Whether to save the processed data to disk (default is False).
    """

    def __config__(self, save_data_path, masks_path):
        """
        Configures paths for processing and saving data.

        Parameters
        ----------
        save_data_path : str
            Path to the directory for saving processed data.
        masks_path : str
            Path to the directory containing masks.

        Returns
        -------
        None
        """

        print("Configuring LION Data Processor...")

        self.save_data_path = save_data_path
        self.masks_path = masks_path

        # Input paths
        self.lipid_programs_hierarchy_path = os.path.join(self.save_data_path, "LipidProgramsHierarchy.csv")
        self.hierarchy_dict_path = os.path.join(save_data_path, "hierarchy_dict.json")
        self.association_path = os.path.join(self.save_data_path, "LION_association.txt")
        self.paths_path = os.path.join(self.save_data_path, "LION.csv")
        self.multi_index_list_path = os.path.join(self.save_data_path, "multi_index_list.pkl")

        self.leaf_level_0_paths_dict_path = os.path.join(
            self.save_data_path, "leaf_level_0_paths_dict.json"
        )
        self.leaf_level_1_paths_dict_path = os.path.join(
            self.save_data_path, "leaf_level_1_paths_dict.json"
        )
        self.leaf_paths_dict_path = os.path.join(self.save_data_path, "leaf_paths_dict.json")

        # Output paths
        self.name_code_mapping_path = os.path.join(
            self.save_data_path, "LION_Name_Code_Mapping.txt"
        )
        self.lipid_programs_path = os.path.join(self.save_data_path, "LION_Lipid_Programs.txt")
        self.lipids_path = os.path.join(self.save_data_path, "LION_Lipids.txt")

        self.paths_curated_path = os.path.join(self.save_data_path, "LION_paths.csv")
        self.lp_parents_path = os.path.join(self.save_data_path, "LP_Parents.csv")

        self.lmt_path = os.path.join(self.masks_path, "LION")

    def __init__(self, save_data_path=".", masks_path="../masks", save_data=False):
        """
        Initializes the LIONDataHandler class and preprocesses the LION database.

        Parameters
        ----------
        save_data_path : str, optional
            Path to the directory for saving processed data (default is './data').
        masks_path : str, optional
            Path to the directory containing masks (default is './masks').
        save_data : bool, optional
            Whether to save the processed data to disk (default is False).
        """

        self.save_data = save_data

        self.__config__(save_data_path, masks_path)

        # Load lipid programs hierarchy and mappings
        self.lipid_programs_hierarchy = pd.read_csv(self.lipid_programs_hierarchy_path, header=None)
        self.lipid_programs_hierarchy.columns = [
            "LION_Program_Name",
            "LION_Program_Code",
        ]
        with open(self.hierarchy_dict_path, "r") as file:
            self.hierarchy_dict = json.load(file)

        self.name_code_mapping = pd.read_csv(
            self.paths_path,
        )
        self.name_code_mapping = self.name_code_mapping[['Preferred Label', 'Class ID']]
        self.name_code_mapping.columns = ["LION_Name", "LION_Code"]
        self.name_code_mapping['LION_Code'] = self.name_code_mapping['LION_Code'].apply(lambda x: x.split("/")[-1].replace("_", ":"))

        # Process database
        self._preprocess_data_()
        self._preprocess_hierarchy_()

    def _preprocess_data_(self):
        """
        Preprocesses the data by adding missing lipid programs (LPs) to the
        LION Name-Code Association database and separating lipid programs and
        lipids into distinct datasets.

        Returns
        -------
        None
        """

        print("\nPreprocessing Data...")

        # Adding missing LPs to the LION Name-Code Association
        print(" -- Adding Missing LPs to the LION Name-Code Association -- ")

        # In the LION_association.txt file there are lipids and the corresponding LION_Code,
        # but not LPs (that still have a LION_Code associated)
        # Moreover, in LION_association.txt there are different names for the same lipid,
        # hence some codes are duplicated.

        new_mapping = pd.DataFrame(columns=["LION_Name", "LION_Code"])

        lp_names = set()
        lp_codes = set(self.lipid_programs_hierarchy["LION_Program_Code"])

        for (
            _,
            row,
        ) in self.lipid_programs_hierarchy.iterrows():  # LipidProgramsHierarchy.csv
            program_name = row["LION_Program_Name"]
            program_name = program_name.replace("-- ", "")
            lp_names.add(program_name)

            program_code = row["LION_Program_Code"]

            code = self.name_code_mapping[self.name_code_mapping["LION_Name"] == program_name][
                "LION_Code"
            ].values

            if len(code) == 0:
                new_row = {"LION_Name": program_name, "LION_Code": program_code}
                new_mapping = pd.concat([new_mapping, pd.DataFrame([new_row])], ignore_index=True)

        self.name_code_mapping = pd.concat([self.name_code_mapping, new_mapping], ignore_index=True)

        if self.save_data:
            self.name_code_mapping.to_csv(
                self.name_code_mapping_path, sep="\t", index=False, header=False
            )

        # Splitting LPs and Lipids in the LION Name-Code Association Database
        print(" -- Splitting LPs and Lipids in the LION Name-Code Association Database -- ")

        # Lipid Programs
        mask_lipid_programs = self.name_code_mapping["LION_Code"].isin(
            lp_codes
        ) & self.name_code_mapping["LION_Name"].isin(lp_names)
        self.lipid_programs = self.name_code_mapping[mask_lipid_programs]
        print(f"Lipid Programs in the LION Database: {self.lipid_programs.shape[0]}")

        if self.save_data:
            self.lipid_programs.to_csv(
                self.lipid_programs_path, sep="\t", index=False, header=False
            )

        # Lipids
        mask_lipids = ~self.name_code_mapping["LION_Code"].isin(lp_codes)
        self.lipids = self.name_code_mapping[mask_lipids]

        # Manage duplicates from different databases (e.g. LIPID MAPS, SwissLipids, etc.)
        # TODO: check if this is still necessary. it should not be necessary if the LION database is updated (LION.csv or LION_Lipids.txt)
        # mask_duplicates = self.lipids["LION_Code"].duplicated(keep=False)
        # duplicates = self.lipids[mask_duplicates]
        # singles = self.lipids[~mask_duplicates]

        # other_nomenclatures = ["SLM:", "LMGP", "LMPR", "LMST", "LMSP", "LMGL"]
        # regex = "|".join(other_nomenclatures)
        # mask_wrong_duplicates = duplicates["LION_Name"].str.contains(regex)
        # mask_wrong_singles = singles["LION_Name"].str.contains(regex)
        # no_more_duplicates = duplicates[~mask_wrong_duplicates]

        # still_singles = singles[~mask_wrong_singles]
        # assert still_singles.shape[0] == singles.shape[0]

        # self.lipids = self.lipids[
        #     (self.lipids["LION_Name"].isin(no_more_duplicates["LION_Name"]))
        #     | (self.lipids["LION_Name"].isin(still_singles["LION_Name"]))
        # ]
        print(f"Lipids in the LION Database: {self.lipids.shape[0]}")

        if self.save_data:
            self.lipids.to_csv(self.lipids_path, sep="\t", index=False, header=False)

    def _preprocess_hierarchy_(self, write=False):
        """
        Extracts hierarchical lipid structure (leaf-root paths) from the LION dataset.
        The hierarchical structure is represented as a path from the leaf to the root.
        The leaf is a given lipid, the root is the most general category and each node
        of the tree can potentially have multiple parents.

        Parameters
        ----------
        write : bool, optional
            Whether to compute the dictionaries (default is False).

        Returns
        -------
        None
        """

        print("\nExtracting Lipids' Hierarchical Structure (leaf-root path)...")

        # Load the LION dataset (~63K lipids) and their hierarchical structure
        self.df_paths = pd.read_csv(self.paths_path)  # LION.csv

        self.df_paths = self.df_paths[
            [
                "Preferred Label",
                "http://www.w3.org/2004/02/skos/core#notation",
                "Parents",
            ]
        ]
        self.df_paths.columns = ["LION_Name", "LION_Code", "Parents"]

        for i in range(0, self.df_paths.shape[0]):
            try:
                parents = self.df_paths.iloc[i]["Parents"].split("|")
            except AttributeError:
                parents = []
            parents = [p.split("/")[-1].replace("_", ":") for p in parents]
            self.df_paths.at[i, "Parents"] = parents
        if self.save_data:
            self.df_paths.to_csv(self.paths_curated_path, index=False)

        print(
            "Splitting Lipid Programs, 0-Step-Lipids and 1-Step-Lipids (this may take a while!)..."
        )

        df_lipids_paths = pd.merge(
            self.df_paths, self.lipids, on=["LION_Code", "LION_Name"], how="inner"
        )
        
        df_lipid_programs_paths = pd.merge(
            self.df_paths,
            self.lipid_programs,
            on=["LION_Code", "LION_Name"],
            how="inner",
        )
        
        # Select only the lipids whose parents are in the set of lipid programs
        # This choice is to process the most comprehensive set of lipids
        def has_all_parents_LPs(S, p):
            return set(p).issubset(S)

        LPs = set(df_lipid_programs_paths["LION_Code"])
        mask = df_lipids_paths["Parents"].apply(lambda p: has_all_parents_LPs(LPs, p))

        df_lipids_paths_level_0 = df_lipids_paths[mask]
        lipids_paths_next = df_lipids_paths[~mask]

        # TODO: manage loading or writing the dictionaries
        # TODO: these dictionaries do not depend on input data, so they can be loaded from disk (computed only the first time)
        if write:
            leaf_level_0 = set(df_lipids_paths_level_0["LION_Code"].values)
            self.leaf_level_0_paths_dict = {leaf: self.__build_dict(leaf) for leaf in leaf_level_0}
            # with open(self.leaf_level_0_paths_dict_path, 'w') as f:
            #     json.dump(self.leaf_level_0_paths_dict, f)

            leaf_level_1 = set(lipids_paths_next["LION_Code"].values)
            self.leaf_level_1_paths_dict = {leaf: self.__build_dict(leaf) for leaf in leaf_level_1}
            # with open(self.leaf_level_1_paths_dict_path, 'w') as f:
            #     json.dump(self.leaf_level_1_paths_dict, f)

            all_leaf = set(df_lipids_paths["LION_Code"].values)
            self.leaf_paths_dict = {leaf: self.__build_dict(leaf) for leaf in all_leaf}
            # with open(self.leaf_paths_dict_path, 'w') as f:
            #     json.dump(self.leaf_paths_dict, f)

        else:
            with open(self.leaf_level_0_paths_dict_path, "r") as f:
                self.leaf_level_0_paths_dict = json.load(f)

            with open(self.leaf_level_1_paths_dict_path, "r") as f:
                self.leaf_level_1_paths_dict = json.load(f)

            with open(self.leaf_paths_dict_path, "r") as f:
                self.leaf_paths_dict = json.load(f)

        print("Splitting Lipid Programs into BiologicalParents and StepParents ...")
        # Splitting Lipid Programs into BiologicalParents (bp) and StepParents (sp)
        self.df_lipid_programs_bpsp = self.__split_LP_parents(df_lipid_programs_paths)

        if self.save_data:
            self.df_lipid_programs_bpsp.to_csv(self.lp_parents_path, index=False)

    def __build_dict(self, node):
        """
        Recursively builds a dictionary representing the hierarchical structure of lipids.

        Parameters
        ----------
        node : str
            The LION_Code of the current lipid or lipid program.

        Returns
        -------
        dict or str
            A nested dictionary representing the parent-child relationships for the node,
            or 'root' if the node has no parents or the parents are root-level ('owl#Thing').
        """
        parents = self.df_paths[self.df_paths["LION_Code"] == node]["Parents"].values[0]
        if not parents or parents == ["owl#Thing"]:
            return "root"
        else:
            return {parent: self.__build_dict(parent) for parent in parents}

    def __get_keys(self, hierarchy_dict, current_node, parent_key=None):
        """
        Retrieves all parent keys associated with a given node in the hierarchy.

        Parameters
        ----------
        hierarchy_dict : dict
            The hierarchy dictionary representing the relationships between nodes.
        current_node : str
            The node for which the parents are being searched.
        parent_key : str, optional
            The current parent key being checked during recursion (default is None).

        Returns
        -------
        list
            A list of parent keys associated with the current node.
        """
        parents = []
        for key, value in hierarchy_dict.items():
            if key == current_node:
                parents.append(parent_key)
            elif isinstance(value, list) and current_node in value:
                parents.append(key)
            elif isinstance(value, dict):
                parents.extend(self.__get_keys(value, current_node, parent_key=key))
        return parents

    def __split_LP_parents(self, df_lipid_programs_paths):
        """
        Splits the lipid programs into their respective biological parents (bp) and step parents (sp), based on hierarchy.
        - Biological Parents are those LPs that, given the current LP, are parents in the tree and share the same macrocategory.
        - Step Parents are those LPs that, given the current LP, are parents in the tree and NOT share the same macrocategory.

        Parameters
        ----------
        df_lipid_programs_paths : pandas.DataFrame
            Dataframe containing LION lipid programs with columns 'LION_Code', 'LION_Name', and 'Parents'.

        Returns
        -------
        pandas.DataFrame
            A modified dataframe where:
            - 'BioParents' column contains only the biological parent node from the hierarchy.
            - 'StepParents' column contains step-parents, if any.
        """

        copy_df = df_lipid_programs_paths.copy()
        copy_df["StepParents"] = copy_df["Parents"]
        copy_df["BioParent"] = copy_df["Parents"]
        for i, row in copy_df.iterrows():
            node = row["LION_Code"]
            parents = self.__get_keys(self.hierarchy_dict, node)
            copy_df.at[i, "StepParents"] = [
                elem for elem in row["StepParents"] if elem not in parents
            ]
            copy_df.at[i, "BioParent"] = parents

        copy_df = copy_df.drop(columns=["Parents"])

        return copy_df

    def _get_LP_BioParent(self, LION_LP_Code):
        """Retrieve the biological parent of a lipid program node given its LION code."""
        parents = self.df_lipid_programs_bpsp[
            self.df_lipid_programs_bpsp["LION_Code"] == LION_LP_Code
        ]["BioParent"].values[0]
        return parents

    def _get_LP_StepParents(self, LION_LP_Code):
        """Retrieve the step-parents of a lipid program node given its LION code."""
        sp = self.df_lipid_programs_bpsp[self.df_lipid_programs_bpsp["LION_Code"] == LION_LP_Code][
            "StepParents"
        ].values[0]
        return sp

    def _get_macrocategory(self, current_node):
        """Recursively retrieve the top-level macrocategory for a given node in the hierarchy."""
        current_parents = self.__get_keys(self.hierarchy_dict, current_node)
        if not current_parents or current_parents[0] is None:
            return current_node
        else:
            for curr_parent in current_parents:
                return self._get_macrocategory(curr_parent)

    def _find_marker_nodes(self, current_node, marker_nodes, visited):
        """
        Recursively traverses the hierarchical structure to identify and mark nodes that represent key
        transitions in the hierarchy, referred to as "marker nodes".

        Parameters
        ----------
        current_node : any
            The starting node for the current traversal in the hierarchy.
        marker_nodes : set
            A set to store the identified categorical transition nodes.
        visited : set
            A set to track nodes that have already been visited to prevent redundant traversal.

        Steps:
        -----
        1. Retrieves the parent node and validate consistency.
        2. Marks the current node as visited and recursively explores:
            a. The parent node to move up the hierarchy.
            b. The step-parents to identify bifurcations at the current level.
        3. If the current node has step-parents, they are added to the `marker_nodes` set.
        4. Repeats the process for each step-parent recursively.

        Notes:
        -----
        - The function ensures that bifurcations or key transitions are identified and marked at the
        appropriate level in the hierarchy.
        - It assumes the hierarchy is a tree-like structure where each node has a unique parent.

        Returns
        --------
        None (modifies `marker_nodes` and `visited` in place).
        """

        # Get the biological parent and step parents of the current node
        current_bp = self._get_LP_BioParent(current_node)
        current_sps = self._get_LP_StepParents(current_node)

        assert len(current_bp) == 1
        current_bp = current_bp[0]
        if current_bp is None:
            return
        macro_cat_node = self._get_macrocategory(current_node)
        macro_cat_parent = self._get_macrocategory(current_bp)
        assert macro_cat_node == macro_cat_parent

        # Recursive scenarios
        if current_node not in visited:
            visited.add(current_node)

            if current_sps != []:  # Bifurcation Node
                marker_nodes.update(current_sps)

            self._find_marker_nodes(current_bp, marker_nodes, visited)

            for current_sp in current_sps:
                self._find_marker_nodes(current_sp, marker_nodes, visited)

    def _process_marker_nodes(self, leaf_level='level_0'):
        """
        The function iterates over all lipids in the database and uses `_find_marker_nodes` to
        recursively identify nodes where a transition or bifurcation occurs between macrocategories.

        Parameters
        ----------
        None (uses instance attributes such as `self.leaf_level_0_paths_dict`).

        Returns
        --------
        all_marker_nodes : dict
            A dictionary where each key is a lipid, and the value is a set of marker LPs for that lipid.

        Notes:
        -----
        - The function ensures that all potential bifurcations and category-defining nodes are accounted
        for at all levels of the hierarchy.
        - First-level nodes are automatically included as they represent the primary step in the tree.
        - For a given lipid hierarchy, the function identifies nodes that cause a jump in categorization
        at higher levels or maintain unique classifications.

        Returns
        --------
        all_marker_nodes : dict
            Dictionary mapping each lipid to its corresponding set of transition nodes.
        """
        assert leaf_level in ['', 'level_0', 'level_1']
        paths_dict = self.leaf_level_0_paths_dict if leaf_level == 'level_0' else self.leaf_level_1_paths_dict if leaf_level == 'level_1' else self.leaf_paths_dict
                
        df_lipid_programs_paths = pd.merge(
            self.df_paths,
            self.lipid_programs,
            on=["LION_Code", "LION_Name"],
            how="inner",
        )
        LPs = set(df_lipid_programs_paths["LION_Code"])

        all_marker_nodes = {}

        for lipid, paths in paths_dict.items(): # self.leaf_level_0_paths_dict, self.leaf_level_1_paths_dict
            if paths == "root":
                continue

            # first_nodes_list = list(paths.keys())
            first_nodes = set(paths.keys()).intersection(LPs)
            # assert len(first_nodes) == len(first_nodes_list)
            marker_nodes = first_nodes.copy()

            for start_node in first_nodes:
                visited = set()
                self._find_marker_nodes(start_node, marker_nodes, visited)
            all_marker_nodes[lipid] = marker_nodes

        return all_marker_nodes

    def _get_level(self, data, marker_nodes):
        """
        Identify the highest hierarchical level in the multi-index columns where each marker node first appears.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame with multi-indexed columns.
        marker_nodes : list
            A list of marker node identifiers to locate in the column hierarchy.

        Returns
        -------
        levels : list
            A list of hierarchical levels (0-indexed) corresponding to the first occurrence of each marker node.
            If a marker node is not found, its corresponding entry will be `None`.
        """
        # check the first occurence of each marker node in the columns
        # the first occurence allows to identify the highest level where the marker node first appears
        # the following levels are neglected as the marker node may appear later in the hierarchy
        # if the path "stops"
        levels = [
            next(
                (
                    level
                    for level in range(data.columns.nlevels)
                    if (data.columns.get_level_values(level) == curr_mn).any()
                ),
                None,
            )
            for curr_mn in marker_nodes
        ]
        return levels

    def _prepare_multi_index_df(self):
        """
        Load and process the multi-index structure for the DataFrame, ensuring all levels are filled.

        Returns
        -------
        data : pd.DataFrame
            A DataFrame with processed multi-index columns where `None` values are replaced
            by propagating the last non-`None` value in the hierarchy.
        """
        # TODO: dynamycally create the multi_index_list, without saving it to memory
        with open(self.multi_index_list_path, "rb") as f:
            multi_index_list = pickle.load(f)
        self.multi_index_list_not_none = [
            tuple(accumulate(tup, lambda last, curr: curr if curr is not None else last))
            for tup in multi_index_list
        ]
        level_names = [
            "MacroCategory",
            "MicroCategory",
            "Level 1",
            "Level 2",
            "Level 3",
            "Level 4",
        ]
        multi_index_not_none = pd.MultiIndex.from_tuples(
            self.multi_index_list_not_none, names=level_names
        )
        data = pd.DataFrame(columns=multi_index_not_none)
        return data

    def tabular_LION_database(self):
        """
        Populate the tabular version of the LION database with marker node presence masks for each lipid.

        Returns
        -------
        data : pd.DataFrame
            A DataFrame where each row corresponds to a lipid, and each column corresponds to a LP
            and rows are filled such that they reporesent the membership of the current lipid to the
            correspinding LPs, according to LION database. The dataframe is multi-indexed, according
            to the hierarchical structure of LION.

        Notes:
        ------
        - For each lipid and its associated marker nodes:
            - Identifies the highest hierarchical levels for the marker nodes using `_get_level`.
            - Creates a mask indicating the presence of each marker node in the corresponding level.
            - Populates the DataFrame row with the mask for the given lipid.
        - The marker node presence is represented as binary values (0 or 1) in the mask.
        """

        data = self._prepare_multi_index_df()
        self.all_marker_nodes = self._process_marker_nodes()

        for lipid_name, curr_marker_nodes in tqdm(self.all_marker_nodes.items()):
            levels = self._get_level(data, curr_marker_nodes)

            mask = np.zeros(len(data.columns), dtype=int)
            for i, curr_mn in enumerate(curr_marker_nodes):
                mask |= (data.columns.get_level_values(levels[i]) == curr_mn).astype(int)

            data.loc[lipid_name, :] = mask

        return data

    def filter_and_aggregate(
        self,
        data,
        refinement_level=2,
        macro_categories=[
            "CAT:0012008",  # cellular component
            "CAT:0012007",  # function
            "CAT:0000000",  # lipid classification
            "CAT:0000091",
        ],  # physical or chemical properties
        micro_categories=[
            "CAT:0000092",  # charge headgroup
            "CAT:0000100",  # contains fatty acid
            "CAT:0000123",  # type by bond
            "CAT:0000463",  # intrinsic curvature
            "CAT:0002945",  # fatty acid unsaturation
            "CAT:0002946",  # fatty acid chain length
            "CAT:0080950",  # lateral diffusion
            "CAT:0080951",  # bilayer thickness
            "CAT:0001734",  # chain-melting transition temperature
            "CAT:0080949",  # tail order parameter
            "CAT:0080948",  # area compressibility
            "CAT:0080947",
        ],  # area per lipid
    ):
        """
        Filters and aggregates the tabular LION database (obtained from `tabular_LION_database()`)
        based on specified macro and micro categories, and a given refinement level. T
        he output is a DataFrame whose columns represent unique LPs at the given level,
        rows correspond to one lipid each and values represent the membership of a lipid to a LP.
        The tabular version of the LION database is built such that the sum of the data across
        all columns at a given refinement level is consistent with the classification of lipids
        (i.e. 0 or 1).

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame with multi-indexed columns, where the first level represents
            macro categories, the second level may represent micro categories,
            and subsequent levels refine these categories.
        refinement_level : int, optional
            The level of the hierarchy (e.g., 'Level 2') to aggregate the data on.
            Defaults to 2.
        macro_categories : list of str, optional
            A list of macro category identifiers to include in the filtering.
            Defaults to:
            - 'CAT:0012008': cellular component
            - 'CAT:0012007': function
            - 'CAT:0000000': lipid classification
            - 'CAT:0000091': physical or chemical properties
        micro_categories : list of str, optional
            A list of micro category identifiers, which are subcategories of
            'CAT:0000091' (physical or chemical properties), to include in the filtering.
            Defaults to:
            - 'CAT:0000092': charge headgroup
            - 'CAT:0000100': contains fatty acid
            - 'CAT:0000123': type by bond
            - 'CAT:0000463': intrinsic curvature
            - 'CAT:0002945': fatty acid unsaturation
            - 'CAT:0002946': fatty acid chain length
            - 'CAT:0080950': lateral diffusion
            - 'CAT:0080951': bilayer thickness
            - 'CAT:0001734': chain-melting transition temperature
            - 'CAT:0080949': tail order parameter
            - 'CAT:0080948': area compressibility
            - 'CAT:0080947': area per lipid

        Returns
        -------
        aggregated_data : pd.DataFrame
            A DataFrame filtered and aggregated at the specified refinement level.

        Raises:
        -------
        AssertionError:
            - If `refinement_level` exceeds 4.
            - If `micro_categories` are provided but 'CAT:0000091' is not in `macro_categories`.
        """
        if not isinstance(macro_categories, list):
            macro_categories = [macro_categories]
        if not isinstance(micro_categories, list):
            micro_categories = [micro_categories]
        assert refinement_level <= 4, "Refinement level should be maximum 4"

        if "CAT:0000091" not in macro_categories:
            micro_categories = []

        if micro_categories != []:
            assert (
                "CAT:0000091" in macro_categories
            ), "Physical or chemical properties should be included in macro_categories"

        cols_to_keep = []
        for col in data.columns:
            if col[0] in macro_categories and col[0] != "CAT:0000091":
                cols_to_keep.append(col)
            elif col[0] == "CAT:0000091" and col[1] in micro_categories:
                cols_to_keep.append(col)
            else:
                continue

        filtered_data = data.loc[:, cols_to_keep]
        level_index = f"Level {refinement_level}"
        aggregated_data = filtered_data.groupby(axis=1, level=level_index).sum()

        assert aggregated_data.values.all() in {
            0,
            1,
        }, "The sum of the data across all columns should be either 0 or 1"

        return aggregated_data

    def _to_LION(self, lipid_names):
        """
        Converts lipid names to LION format (the LIPID MAPS format).
        """

        def process_name(name):
            # Remove specific substrings
            for substring in ["(OH)", ";O2", ";O3", ";O4", ";O"]:
                name = name.replace(substring, "")
                # Add '/0:0' for names starting with 'L' in the first word
            if name.split(" ")[0].startswith("L"):
                name = name[1:] + "/0:0"
            if name.split(" ")[0]=="FA":
                name = "F" + name
            # Replace space with '(' and append ')'
            return name.replace(" ", "(") + ")"

        return [process_name(name) for name in lipid_names]

    def create_ontology_lmt(self, lipid_names, lion_tab_db, save_name="full"):
        """
        Creates a `.lmt` (Lipid Matrix Transposed) file based on lipid ontology and input data,
        mapping lipid names to their corresponding codes and associating them with lipid programs.

        Parameters
        ----------
        lipid_names : list
            A list of lipid names corresponding to the rows of the `data` DataFrame.
        lion_tab_db : pd.DataFrame
            A binary DataFrame where:
            - Rows represent lipids (indexed by lipid codes).
            - Columns represent lipid programs (indexed by program codes).
            - A value of `1` indicates the lipid is associated with the lipid program.
        save_name : str, optional
            The base name for the output `.lmt` file. The file will be saved as
            `<self.lmt_path>_<save_name>.lmt`. Defaults to 'full_LION'.

        Returns
        -------
        None
            Writes the `.lmt` file to the specified path (`self.lmt_path`).

        File Format:
        ------------
        The `.lmt` file is a tab-delimited file where each line represents a lipid program, followed by
        the names of the lipids associated with that program. Example:
        ```
        Program_Name    Lipid_Name_1    Lipid_Name_2    Lipid_Name_3
        ```

        """
        
        self.matches = pd.DataFrame(
            {"LBA_Name": lipid_names, "LION_Name": self._to_LION(lipid_names)}
        )
        
        matches_dict = (
            self.matches.groupby("LION_Name")
            .apply(lambda g: g["LBA_Name"][g["LBA_Name"].str.len().idxmin()])
            .to_dict()
        )
        lipid_codes_to_names = (
            pd.read_csv(self.lipids_path, sep="\t", header=None, index_col=1, squeeze=True)
            .apply(lambda v: matches_dict.get(v, v))
            .to_dict()
        )
        program_codes_to_names = pd.read_csv(
            self.lipid_programs_path, sep="\t", header=None, index_col=1, squeeze=True
        ).to_dict()
        
        with open(self.lmt_path + "_" + save_name + ".lmt", "w") as f:
            for column in lion_tab_db.columns:
                program_name = program_codes_to_names.get(column, column)
                rows = [
                    lipid_codes_to_names.get(index, index)
                    for index in lion_tab_db[lion_tab_db[column] == 1].index
                ]
                f.write(
                    f"{program_name}\t"
                    + "\t".join([str(row).replace("_", " ") for row in rows])
                    + "\n"
                )