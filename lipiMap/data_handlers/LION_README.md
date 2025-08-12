# LIONDataHandler Documentation

## LION - Lipid Ontology
The [Lipid Ontology (LION)](https://bioportal.bioontology.org/ontologies/LION) database and the corresponding web tool, [LION/web](http://www.lipidontology.com/), serve as a comprehensive platform for integrating lipid data with biological functions, chemical characteristics, and cellular components. LION categorizes over *50,000 lipid species* into four primary branches:

| Category                | Description                                                                                                                                                             |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lipid Classification**| Each lipid is categorized under one of the following: fatty acids, glycerolipids, glycerophospholipids, sphingolipids, sterol lipids, prenol lipids, saccharolipids, polyketides.                                      |
| **Function**            | Lipids are assigned functions such as lipid-mediated signalling, membrane component, and lipid storage.                                                                  |
| **Cellular Component**| Lipids are associated with one cellular component, including nucleus, endoplasmic reticulum, mitochondrion, plasma membrane, chloroplast, lipid droplet, Golgi apparatus, endosome/lysosome, peroxisome.          |
| **Physico-Chemical Properties**| Lipids may have properties from subcategories: charge headgroup, fatty acid composition, type by bond, intrinsic curvature, fatty acid unsaturation, fatty acid chain length, lateral diffusion, bilayer thickness, chain-melting transition temperature. |

### Input Data
Below is a table describing the input data files required by the `LIONDataHandler` class, along with their formats and descriptions:

| File Name                   | Format                  | Description                                                                                                                         |
|-----------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **LION_association.txt**    | Tab-separated TXT file  | Contains each lipid name and the associated LION code. Each line represents a unique association between a lipid and its LION code. |
| **LION.csv**                | CSV file                | Lists all lipids as rows, featuring crucial information such as the lipid's LION code and its parents in the hierarchical structure. |
| **LipidProgramsHierarchy.csv** | CSV file            | Provides a nested representation of macro categories, micro categories, and lipid programs, illustrating the hierarchical organization of lipid categorizations. |
| **multi_index_list.pkl**    | Pickle (`.pkl`) file    | Serialized Python object that stores multi-index structures needed for organizing data in the DataFrame according to the LION hierarchy. |

**Usage Notes:**

- **LION_association.txt** and **LION.csv** are essential for mapping lipids to their respective codes and understanding their placement within the hierarchical framework of the LION database.
- **LipidProgramsHierarchy.csv** is crucial for operations that require knowledge of the nested categorization of lipids, such as filtering and aggregation based on specific categories or levels.
- **multi_index_list.pkl** is used to rebuild the DataFrame structure used in the analysis, ensuring that the multi-indexed columns correctly represent the hierarchical levels specified in the LION data.


### Hierarchical Structure
Similarly to other ontological frameworks in biological sciences, such as the Gene Ontology, LION employs a structured methodology to organize lipid data into these categories. Indeed, one of the significant advantages of the LION database is its ability to generate hierarchical networks that trace pathways from a specific lipid to its four main categories.

![LION Hierarchical Structure](./images_for_thesis/lion_tree_PS.png)

*Figure: Hierarchical Tree of PS 45:7. This illustration depicts the structured categorization of the lipid, with each category represented by a unique color — blue for physical and chemical properties, pink for lipid families, orange for cellular components, and green for functions. At the top of the hierarchy is the lipid to be analyzed: the deeper one walks the hierarchy, the coarser the classification.*

Based on this hierarchical structure, LION provides 406 possible classes that can serve as programs when working with the complete lipidomic database. It is important to note that some overlap with the Lipid Families (see `LBADataHandler`) may occur, but this is addressed through collinearity removal. The LION database includes over *250,000* potential connections.

#### How to extract Lipid Programs from LION database?

##### **`tabular_LION_database()`** 
This method is designed to construct a tabular representation of the LION database. This table maps each lipid to its corresponding Lipid Programs (LPs) based on hierarchical classifications and relationships defined within the LION database. Below is a schematic overview of the method and its key interactions:

1. **DataFrame Preparation**:
    - The method begins by setting up a DataFrame with a columns' multi-index structure that represents various hierarchical levels of lipid categorization.
    - This structure allows for a detailed classification and placement of each lipid based on its hierarchical relationships within the database. It is obtained straightforwardly from the `LipidProgramsHierarchy.csv` file given as input to the class.
    - The multi-index results in a 6-levels structure, where:
        - the first 2 levels are for Macro (Lipid Classification, Function, Cellular Component and Physico-Chemical Properties) and Micro (Lipid Classification, Function, Cellular Component and the subcategories of PC properties: charge headgroup, fatty acid composition, type by bond, intrinsic curvature, fatty acid unsaturation, fatty acid chain length, lateral diffusion, bilayer thickness, chain-melting transition temperature) categories.
        - the remaining 4 levels are the classes (i.e. the lipid programs) that belong to the aforementioned categories.

    ![Header MultiIndex](./images_for_thesis/multi_index.png)
    
    *Figure: The figure illustrates the multi-index structure of the DataFrame representing the LION database in tabular format. For readability, columns are arranged over three lines, though in practice, they appear on a single line. The first two levels of the multi-index correspond to categories and macrocategories, denoted with "CAT." While macrocategories are labeled in the figure, subcategories for physico-chemical properties are identifiable via codes listed in `LipidProgramsHierarchy.csv`. Following these are four levels detailing LPs, with mappings also available in `LipidProgramsHierarchy.csv`. This layered structure underscores the utility of the multi-index format.*

2. **Marker Nodes Identification**:
    - **Marker nodes**, which are key hierarchical transitions or bifurcation points in the lipid ontology, are identified for each lipid (`_find_marker_nodes` and `_process_marker_nodes`). This identification is crucial for understanding how lipids relate to different LPs.
    - To identify marker nodes, for each given node in the tree-like hierarchy, we identify **BiologicalParents** and **StepParents** (`__split_LP_parents()`), respectively those that belong to the same category, and those that instead belong to a different category. This is the *bifurcation* we refer to. The identification process involves recursively traversing the lipid hierarchy and marking transitions where lipid categorizations change or bifurcate.
    - Therefore, marker nodes are all those LPs from which a *path-back* to the main categories is uniquely defined. The figure depicts a representative example.
    - For each lipid, the method identifies the presence of marker nodes at various levels of the hierarchy and creates a binary membership mask. Specifically, marker nodes represent the *ones*. For each lipid *i* and lipid program *j*, the membership mask is 1 if lipid *i* belongs to LP *j*. This mask is then used to populate the DataFrame, indicating the association of each lipid with various LPs based on the presence or absence of specific marker nodes.

    ![Marker Nodes](./images_for_thesis/lion_marker_nodes.png)
    
    *Figure: The figure presents the hierarchical structure for PS 45:7, highlighting marker nodes which characterize distinct lipid programs. Bold boxes indicate marker nodes, with dashed lines depicting "step-parenthood" and solid lines showing "biological-parenthood". Each marker node distinctly traces a path to its respective category, ensuring no ambiguity.*

##### **Interaction with the tabular LION database**:
Once the tabular version of the database is built, the user can interact with it at the level they want.
The `filter_and_aggregate()` method is designed to process the tabular representation of the LION database by filtering and aggregating data based on the user's needs.

1. **Filtering**: Select columns in the data that match the input macro categories and, if applicable, micro categories. The user inputs the categories they want to keep into account in the analysis (`macro_categories`) and the subcategories for physico-chemical properties (`macro_categories`).
2. **Aggregating**: Summarize the data at a specified refinement level: indeed, the hierarchical structure provides (at most) 4 levels of refinement (increasigly higher level of specificity of LPs). This aggregation is designed to ensure that each lipid's association with categories at this level is clear and binary (0 or 1).

![Extraction of LPs from LION](./images_for_thesis/lion_mat_PS.png)

![Extraction of LPs from LION](./images_for_thesis/tabular_LION.png)

*Figure: Upon selecting a refinement level (here, the specific 4th level marker nodes are shown), each row of the tabular DataFrame is populated. Users also have the option to adjust the cropping and aggregation of columns according to their desired focus category.*

#### Handling more specific annotations
To conclude, an important consideration is the extent to which measurements can be detailed. Typically, lipids are annotated based on their lipid family, the total number of carbon atoms and double bonds across all fatty acyl chains. This general nomenclature, however, does not reveal the distribution of these carbon atoms and double bonds among fatty acyl chains. In contrast, the LION database provides such higher level of specificity. If we consider the *shrinked* case, LION database presents *7250 lipids*, the extended version is istead made up of more than *60000 lipids*. Here, an example of the different nomenclatures and the shrinking procedure.

![Shrinking of LION Database](./images_for_thesis/shrinking_of_lion_PS.png)

*Figure: Shrinkage procedure applied to some lipids of the LION database. The figure show the recurring example of this documentation: PS 45:7.*