import os
import networkx as nx

import matplotlib.pyplot as plt


caseDir = '/Users/projectephantom/Desktop/base_de_dades_marc/dataset102_713/unprocessed/19963073'

graph_path = os.path.join(caseDir, "graph.pickle")

G = nx.readwrite.gpickle.read_gpickle(graph_path)

nodeTypesDict = {
    0: "other",
    1: "endpoint",
    2: "AA_BT",
    3: "AA-LCCA",
    4: "AA-LSA",
    5: "AA-RSA",
    6: "BT-LCCA",
    7: "BT-RCCA/RSA",
    8: "RSA-RVA",
    9: "LSA-LVA",
    10: "RCCA-RICA/RECA",
    11: "LCCA-LICA/LECA",
    12: "RVA/LVA-BA",
    13: "BT-LSA",
    14: "AA-BT/LCCA",
    15: "AA-LVA",
    16: "BT-LSA",
    17: "AA-RCCA",
    18: "LCCA-LSA"
}

edgeTypesDict = {
    0: "other",
    1: "AA",
    2: "BT",
    3: "RCCA",
    4: "LCCA",
    5: "RSA",
    6: "LSA",
    7: "RVA",
    8: "LVA",
    9: "RICA",
    10: "LICA",
    11: "RECA",
    12: "LECA",
    13: "BA",
    14: "AA+BT",
    15: "RVA+LVA"
}

if os.path.isfile(os.path.join(caseDir, "graph_label.pickle")):
    print("Label is already made")
else:
    # for node in G.nodes:
    #     print("Node", node, "connecting edges:")
    #     for edge in G.edges(node):
    #         print("     ", G.edges[edge]["CellID"])
    #     # input("This node's type is: ")
    #     if G.degree(node) == 1:
    #         G.nodes(data=True)[node]["nodetype"] = 1
    #         G.nodes(data=True)[node]["nodeTypeName"] = nodeTypesDict[1]
    #         print("This node's type is: 1")
    #     else:
    #         G.nodes(data=True)[node]["nodetype"] = int(input("This node's type is: "))
    #         G.nodes(data=True)[node]["nodeTypeName"] = nodeTypesDict[G.nodes(data=True)[node]["nodetype"]]

    for n0, n1 in G.edges:
        print("Edge", G[n0][n1]["CellID"])
        G[n0][n1]["edgetype"] = int(input("This edge's type is: "))
        G[n0][n1]["edgeTypeName"] = edgeTypesDict[G[n0][n1]["edgetype"]]

    nx.readwrite.gpickle.write_gpickle(G, os.path.join(caseDir, "graph_label.pickle"), protocol=4)

    node_pos_dict_P = {}
    for n in G.nodes():
        node_pos_dict_P[n] = [G.nodes(data=True)[n]["pos"][0], G.nodes(data=True)[n]["pos"][2]]

    edge_labels = nx.get_edge_attributes(G, 'edgeTypeName')

    nx.draw(G, node_pos_dict_P, node_size=20)
    nx.draw_networkx_edge_labels(G, node_pos_dict_P, edge_labels = edge_labels)

    # nx.draw(G, node_pos_dict_R, node_size=20)
    # nx.draw_networkx_edge_labels(G, node_pos_dict_R, edge_labels = edge_labels)

    plt.savefig(os.path.join(caseDir, "graph_label.png"))
    # plt.close()
    plt.show()