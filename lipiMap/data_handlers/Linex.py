# Imports
import os

from collections import defaultdict

import community as community_louvain
import igraph as ig
import infomap
import leidenalg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from linex2 import LipidNetwork
from scipy.linalg import eigh
from sklearn.cluster import KMeans


### Linex2 Class - ReactionNetwork Handler
# TODO: document this class properly once the Linex2 code is integrated and modified accordinglyto the lab's needs


class LinexDataHandler:
    def __config__(self, masks_path):
        print("Configuring Linex2 Data Processor...")
        self.masks_path = masks_path
        self.lmt_path = os.path.join(self.masks_path, "linex")

    def __init__(self, lipid_names, masks_path="./masks"):
        self.__config__(masks_path)
        self.lipid_names = self._to_Linex(lipid_names)
        print("Lipid names: ", self.lipid_names)

        # Linex Input Data
        self.upload_to_linex = self.create_linex_input_data()

        # TODO: using the Linex code
        self.create_network()

        self.isolated_nodes = [
            node
            for node in self.lipid_network.network.nodes()
            if self.lipid_network.network.degree(node) == 0
        ]

    def _to_Linex(self, lipid_names):
        return [name.replace(" ", "(") + ")" for name in lipid_names]

    def _from_linex(self, lipid_name):
        return lipid_name.replace("(", " ")[:-1]

    def create_linex_input_data(self):
        # TO BE MODIFIED/GENERALIZED
        return pd.DataFrame(columns=self.lipid_names)

    def create_network(self):
        self.lipid_network = LipidNetwork(data=self.upload_to_linex, 
                                          allow_molspec_fails=True)
        self.lipid_network.compute_native_network()

        def add_node_ids(network):
            for i, node in enumerate(network.nodes):
                network.nodes[node]["id"] = i

        add_node_ids(self.lipid_network.network)
        pos = nx.spring_layout(
            self.lipid_network.network, k=0.5, iterations=100
        )  ########### !!! ###########
        for node in self.lipid_network.network.nodes:
            self.lipid_network.network.nodes[node]["pos"] = pos[node]

    def plot_graph(self):
        G = self.lipid_network
        pos = nx.get_node_attributes(G.network, "pos")

        plt.figure(figsize=(20, 20))
        nx.get_node_attributes(G.network, "pos")
        G._add_network_attribute_(
            "lipid_class",
            nodes=True,
        )
        node_discrete = G._discrete_map_["lipid_class"]
        node_colors = G._node_colour_subset_(
            attr="lipid_class", discrete=node_discrete, cmap="tab10"
        )
        nx.draw_networkx_nodes(
            G.network,
            pos,
            node_color=node_colors[1],
            node_size=800,
            alpha=0.8,
            # cmap=plt.cm.tab20,
        )

        G._add_network_attribute_(
            "reaction_type",
            nodes=False,
        )
        edge_discrete = G._discrete_map_["reaction_type"]
        edge_colors = G._edge_colour_subset_(
            "reaction_type",
            edge_discrete,
        )
        nx.draw_networkx_edges(
            G.network,
            pos,
            edge_color=edge_colors[1],
            alpha=0.8,
            width=1.5,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            connectionstyle="arc3, rad=0.1",
        )

        node_labels = nx.get_node_attributes(G.network, "data_name")
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.title("Lipid Reactions Network", fontsize=20)
        plt.axis("off")

        edge_handles = [plt.Line2D([0], [0], color=val, lw=4) for val in edge_colors[0].values()]
        plt.legend(
            edge_handles,
            edge_colors[0].keys(),
            title="reaction types",
            loc="best",
            fontsize=15,
        )
        plt.show()

    def plot_clusters(self, partition, title=None, values=None):
        G = self.lipid_network.network
        pos = nx.get_node_attributes(G, "pos")

        if values is None:
            values = [
                np.mean([G.nodes[node]["id"] for node in component]) for component in partition
            ]
            add_legend = False
        else:
            add_legend = True

        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        ranks = scipy.stats.rankdata(values)
        values = ranks / len(np.unique(values))

        component_value = {
            node: values[idx] for idx, component in enumerate(partition) for node in component
        }
        colors = [plt.cm.rainbow(component_value[node]) for node in G.nodes()]
        pos = nx.get_node_attributes(G, "pos")

        plt.figure(figsize=(20, 20))
        nx.draw_networkx_nodes(
            G, pos, node_color=colors, node_size=800, alpha=0.6, cmap=plt.cm.rainbow
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="black",
            alpha=0.2,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            #    connectionstyle=f'arc3, rad={rad}', # --> to make the edges curved
        )

        node_labels = nx.get_node_attributes(G, "data_name")
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        if add_legend:
            unique_values = np.unique(values)
            node_handles = [
                plt.Line2D([0], [0], color=plt.cm.rainbow(val), lw=4) for val in unique_values
            ]
            plt.legend(
                node_handles,
                ["low", "medium", "high"],
                title="Node values",
                loc="best",
                fontsize=15,
            )

        plt.title(title, fontsize=20)
        plt.axis("off")
        plt.savefig(
            "/home/fventuri/francesca/lipiMap/lipiMap/dataset/_LBA_Data/data/reactions_cc_louvain.png"
        )
        plt.show()

    def connected_components(self):
        # self.connected_components_ = list(nx.connected_components(self.lipid_network.network))
        G = self.lipid_network.network
        non_isolated_components = [
            component for component in nx.connected_components(G) if len(component) > 1
        ]
        self.connected_components_ = [set(self.isolated_nodes)] + non_isolated_components

    def louvain_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph

        partition = community_louvain.best_partition(G)
        components = []

        if graph is None and self.isolated_nodes:
            components.append(set(self.isolated_nodes))

        for cluster in set(partition.values()):
            if list(partition.values()).count(cluster) == 1:
                continue
            components.append(set([k for k, v in partition.items() if v == cluster]))

        return components

    def leiden_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph
        G_IG = ig.Graph.TupleList(list(G.edges()), directed=False)

        components = []
        if graph is None and self.isolated_nodes:
            components.append(set(self.isolated_nodes))
        partition = leidenalg.find_partition(G_IG, leidenalg.RBERVertexPartition)
        components.extend(
            set(G_IG.vs[vertex]["name"] for vertex in cluster) for cluster in partition
        )

        return components

    def infomap_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph
        mapping = {node: i for i, node in enumerate(G.nodes())}
        reversed_mapping = {i: node for node, i in mapping.items()}
        G_IG = ig.Graph(len(G), [(mapping[edge[0]], mapping[edge[1]]) for edge in G.edges()])

        for node in G.nodes(data=True):
            G_IG.vs[mapping[node[0]]].update_attributes(node[1])

        for edge in G.edges(data=True):
            edge_index = G_IG.get_eid(mapping[edge[0]], mapping[edge[1]])
            G_IG.es[edge_index].update_attributes(edge[2])

        G_IG["name"] = reversed_mapping

        infomap_algorithm = infomap.Infomap()
        infomap_algorithm.add_links((edge.source, edge.target) for edge in G_IG.es)
        infomap_algorithm.run()
        clusters = infomap_algorithm.get_modules()

        components = [
            set(v["data_name"] for v in G_IG.vs if clusters.get(v.index) == i)
            for i in set(clusters.values())
        ]
        if graph is None and self.isolated_nodes:
            components.append(set(self.isolated_nodes))
        return components

    def spectral_clustering(self, graph=None, n_clusters=3):
        G = self.lipid_network.network if graph is None else graph
        L = nx.normalized_laplacian_matrix(G, weight=None).astype(np.float64)
        eigenvalues, eigenvectors = eigh(L.todense())

        cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(
            eigenvectors[:, 1:n_clusters]
        )
        components = []
        for comp in [set(tuple(np.where(cluster_labels == i)[0])) for i in range(n_clusters)]:
            node_names = [
                G.nodes[node]["data_name"] for node in G.nodes if G.nodes[node]["id"] in comp
            ]
            components.append(set(node_names))
        return components

    def extract_centrality_components(self, centrality):
        values = np.array(list(centrality.values())).reshape(-1, 1)
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(values)
        cluster_labels = kmeans.labels_
        node_clusters = {node: label for node, label in zip(centrality.keys(), cluster_labels)}

        components = [
            set([node for node, cluster in node_clusters.items() if cluster == i])
            for i in range(n_clusters)
        ]

        avg_score = [np.mean([centrality[node] for node in comp]) for comp in components]

        return components, avg_score

    def degree_centrality_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph
        degree_centrality = nx.degree_centrality(G)
        components, avg_score = self.extract_centrality_components(degree_centrality)
        return components, avg_score

    def betweenness_centrality_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph
        betweenness_centrality = nx.betweenness_centrality(G)
        components, avg_score = self.extract_centrality_components(betweenness_centrality)
        return components, avg_score

    def closeness_centrality_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph
        closeness_centrality = nx.closeness_centrality(G)
        components, avg_score = self.extract_centrality_components(closeness_centrality)
        return components, avg_score

    def reactiontype_clustering(self, graph=None):
        G = self.lipid_network.network if graph is None else graph

        components = {}
        for u, v, data in G.edges(data=True):
            reaction_type = data["reaction_type"]
            if reaction_type not in components:
                components[reaction_type] = set()
            components[reaction_type].add(u)
            components[reaction_type].add(v)

        # components = list(components.values())
        if graph is None and self.isolated_nodes:
            components["Isolated"] = set(self.isolated_nodes)

        return components

    def plot_reactiontypes(self, reactiontype_dict, title=None):
        G = self.lipid_network.network
        pos = nx.get_node_attributes(G, "pos")

        reaction_colors = {
            "L_FAdelete": "blue",
            "L_FAmodify": "orange",
            "L_HGdelete": "green",
            "L_HGmodify": "red",
        }

        for rt_name, rt_nodes in reactiontype_dict.items():
            if rt_name == "Isolated":
                continue
            color_map_nodes = []
            for node in G.nodes():
                if node not in rt_nodes:
                    color_map_nodes.append("lightgrey")
                else:
                    color_map_nodes.append(reaction_colors[rt_name])

            plt.figure(figsize=(20, 20))
            nx.draw_networkx_nodes(G, pos, node_color=color_map_nodes, node_size=800, alpha=0.6)
            nx.draw_networkx_edges(
                G,
                pos,
                edge_color="black",
                alpha=0.2,
                #    arrows=True, arrowstyle='-|>', arrowsize=10,
                #    connectionstyle=f'arc3, rad={rad}', # --> to make the edges curved
            )

            # nx.draw_networkx_edges(G, pos,
            #                    width=edge_widths,
            #                    arrows=True,
            #                    arrowstyle='-|>',
            #                    arrowsize=20,
            #                    connectionstyle=f'arc3, rad={rad}', --> to make the edges curved
            #                    edge_color='red')

            node_labels = nx.get_node_attributes(G, "data_name")
            nx.draw_networkx_labels(G, pos, labels=node_labels)

            plt.title(f"{title} - {rt_name}", fontsize=20)
            plt.axis("off")
            plt.show()

    def transitions_across_families(self):
        self.bridging_families = defaultdict(list)
        G = self.lipid_network.network

        for u in G:
            for v in G[u]:
                for w in G[v]:
                    family_u = G.nodes[u]["lipid_class"]
                    family_w = G.nodes[w]["lipid_class"]
                    if family_u != family_w:
                        # Sort the tuple to ensure (family1, family2) is the same as (family2, family1)
                        sorted_families = tuple(sorted((family_u, family_w)))
                        self.bridging_families[sorted_families].append(
                            (
                                G.nodes[u]["data_name"],
                                G.nodes[v]["data_name"],
                                G.nodes[w]["data_name"],
                            )
                        )

    def plot_transition(self, family1, family2, title=None):
        G = self.lipid_network.network
        first_nodes = set(
            p
            for path in self.bridging_families[(family1, family2)]
            for p in path
            if family1.replace(" ", "(") in p
        )
        intermediate_nodes = set(path[1] for path in self.bridging_families[(family1, family2)])
        final_nodes = set(
            p
            for path in self.bridging_families[(family1, family2)]
            for p in path
            if family2.replace(" ", "(") in p
        )
        transition_nodes = first_nodes.union(intermediate_nodes).union(final_nodes)
        color_map_nodes = []

        for node in G.nodes():
            if node not in transition_nodes:
                color_map_nodes.append("lightgrey")
            else:
                # print(node)
                if G.nodes[node]["lipid_class"] == family1:
                    color_map_nodes.append("orange")
                elif G.nodes[node]["lipid_class"] == family2:
                    color_map_nodes.append("green")
                else:
                    color_map_nodes.append("pink")
                # print(color_map_nodes[-1])

        first_edges = [
            (node[0], node[1]) if node[0] in first_nodes else (node[2], node[1])
            for node in self.bridging_families[(family1, family2)]
        ]
        second_edges = [
            (node[1], node[2]) if node[2] in final_nodes else (node[0], node[1])
            for node in self.bridging_families[(family1, family2)]
        ]
        transition_edges = set(first_edges + second_edges)

        color_map = []
        widths = []
        for edge in G.edges():
            if edge in transition_edges or edge[::-1] in transition_edges:
                widths.append(2)
                if edge in first_edges or edge[::-1] in first_edges:
                    color_map.append("orange")
                elif edge in second_edges or edge[::-1] in second_edges:
                    color_map.append("green")

            else:
                widths.append(0.1)
                color_map.append("lightgrey")

        pos = nx.get_node_attributes(G, "pos")

        plt.figure(figsize=(20, 20))
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=color_map_nodes,
            node_size=700,
            alpha=0.8,
            cmap=plt.cm.rainbow,
        )
        node_labels = nx.get_node_attributes(G, "data_name")
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        nx.draw_networkx_edges(G, pos, edge_color=color_map, width=widths, alpha=0.8)

        plt.title(title, fontsize=20)
        plt.axis("off")
        plt.show()

    def cluster_subgraph(self, components, clustering_method="louvain", **kwargs):
        clusters = [set(self.isolated_nodes)]
        for component in components:
            subgraph = self.lipid_network.network.subgraph(component).copy()
            method = getattr(self, f"{clustering_method}_clustering")
            subclusters = method(graph=subgraph, **kwargs)
            clusters.extend(subclusters)
        return clusters

    def create_reactions_lmt(self, components, save_name="linex_reactions", avg_scores=None):
        """
        Creates a .lmt file based on the reactions from the given components.
        """
        with open(self.lmt_path + "_" + save_name + ".lmt", "w") as f:
            for i, component in enumerate(components):
                if component == set(self.isolated_nodes):
                    continue
                if avg_scores is None:
                    f.write(
                        f"{save_name}_comp_{i}\t"
                        + "\t".join([self._from_linex(str(elem)) for elem in component])
                        + "\n"
                    )
                else:
                    level = (
                        "low"
                        if avg_scores[i] == min(avg_scores)
                        else "high" if avg_scores[i] == max(avg_scores) else "medium"
                    )
                    f.write(
                        f"{save_name}_{level}\t"
                        + "\t".join([self._from_linex(str(elem)) for elem in component])
                        + "\n"
                    )
