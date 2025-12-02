from collections import defaultdict
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def extract_edges_from_trace(csv_path):
    # Load the trace as a DataFrame
    df = pd.read_csv(csv_path)

    # Index spans by spanID to allow parent lookup
    spans = df.set_index("spanID")

    edges = set()

    for span_id, span in spans.iterrows():
        parent_id = span["parentSpanID"]

        # skip root spans
        if pd.isna(parent_id) or parent_id == "":
            continue

        # ensure parent span exists in the dataset
        if parent_id in spans.index:
            parent = spans.loc[parent_id]

            parent_service = str(parent["serviceName"]).lower()
            child_service = str(span["serviceName"]).lower()
            operation = span["methodName"]

            # record only cross-service calls
            if parent_service != child_service:
                edges.add((parent_service, child_service, operation))

    return list(edges)

def build_graph(edges):
    G = nx.DiGraph()
    for src, dst, op in edges:
        G.add_edge(src, dst, operation=op)
    return G

# Compute topological levels (layered left→right layout)
def compute_levels(G):
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            levels[node] = 0
        else:
            levels[node] = max(levels[p] for p in preds) + 1
    return levels

def visualize_graph(G):
    levels = compute_levels(G)

    # Reverse map: level → list of nodes in that level
    layer_nodes = defaultdict(list)
    for node, level in levels.items():
        layer_nodes[level].append(node)

    # Assign x,y positions manually
    pos = {}
    horizontal_spacing = 3  # spacing between levels (left→right)
    vertical_spacing = 2  # spacing between nodes vertically

    for level, nodes in layer_nodes.items():
        # center vertically
        offset = -(len(nodes) - 1) * vertical_spacing / 2
        for i, node in enumerate(nodes):
            x = level * horizontal_spacing
            y = offset + i * vertical_spacing
            pos[node] = (x, y)

    plt.figure(figsize=(12, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        arrows=True,
        font_size=10,
    )
    edge_labels = {(u, v): d.get("operation", "calls") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    plt.title("Left-to-Right DAG Layout (Manual Topological Levels)")
    plt.axis("off")
    plt.show()

def graph_to_text(graph):
    lines = []
    try:
        topo_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Graph is not a DAG; cannot do topological sort")

    for node in topo_order:
        lines.append(f"Node {node}")

    for u, v, data in graph.edges(data=True):
        rel = data.get("operation", "calls")
        lines.append(f"Edge: {u} --({rel})--> {v}")

    return "\n".join(lines)

def main():
    csv_path = "dataset/RE3-OB/cartservice_f1/1/traces.csv"   # your file name
    edges = extract_edges_from_trace(csv_path)

    print("Extracted edges:")
    for e in edges:
        print(e)

    G = build_graph(edges)
    visualize_graph(G)
    print(graph_to_text(G))

if __name__ == '__main__':
    main()
