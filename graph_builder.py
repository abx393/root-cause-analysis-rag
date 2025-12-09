from collections import defaultdict
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import yaml
matplotlib.use('TkAgg')

import re

def extract_service_from_operation(op_name: str) -> str | None:
    """
    Examples:
      hipstershop.AdService/GetAds -> adservice
      grpc.hipstershop.CurrencyService/GetSupportedCurrencies -> currencyservice
    """
    if not isinstance(op_name, str):
        return None

    # match "...<ServiceName>/Method"
    m = re.search(r'([\w\.]+Service)/', op_name)
    if not m:
        return None

    # take last token after dots
    service = m.group(1).split('.')[-1]
    return service.lower()

def extract_edges_from_trace(csv_path):
    df = pd.read_csv(csv_path)
    spans = df.set_index("spanID")

    edges = set()

    for span_id, span in spans.iterrows():
        parent_id = span["parentSpanID"]

        if pd.isna(parent_id) or parent_id == "":
            continue
        if parent_id not in spans.index:
            continue

        parent = spans.loc[parent_id]

        parent_service = str(parent["serviceName"]).lower()
        child_service = str(span["serviceName"]).lower()
        operation = span["operationName"]

        # 1. Observed span-to-span service edge
        if parent_service != child_service:
            edges.add((parent_service, child_service, operation))

        # 2. Operation-inferred outbound call (child → inferred)
        inferred_service = extract_service_from_operation(operation)

        if (
            inferred_service
            and inferred_service != child_service
        ):
            edges.add((child_service, inferred_service, operation))

    return list(edges)

def build_graph(edges):
    G = nx.DiGraph()
    for src, dst, op in edges:
        G.add_edge(src, dst, operation=op)
    return G

def compute_sccs(G):
    """
    Returns:
        scc_list: List of SCCs, each a list of service nodes.
        scc_map: Dict mapping each node -> scc_id
    """
    scc_list = list(nx.strongly_connected_components(G))
    scc_list = [sorted(list(scc)) for scc in scc_list]

    # Map each node to its SCC index
    scc_map = {}
    for i, scc in enumerate(scc_list):
        for node in scc:
            scc_map[node] = ','.join(scc)

    return scc_list, scc_map

def build_scc_dag(G, scc_map):
    """
    Build a DAG where each SCC is a supernode.
    """
    dag = nx.DiGraph()

    # Add SCC nodes
    for scc_id in set(scc_map.values()):
        dag.add_node(scc_id)

    # Add edges between SCC nodes when cross-SCC deps exist
    for u, v, d in G.edges(data=True):
        scc_u = scc_map[u]
        scc_v = scc_map[v]

        if scc_u != scc_v:
            dag.add_edge(scc_u, scc_v, operation=d.get('operation', 'calls'))

    return dag

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

def visualize_graph_dag(dag):
    levels = compute_levels(dag)

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
        dag,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        arrows=True,
        font_size=10,
    )
    edge_labels = {(u, v): d.get("operation", "calls") for u, v, d in dag.edges(data=True)}
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_size=6)

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

def serialize_scc_dag_to_yaml(scc_dag):
    """
    Serialize an SCC DAG to YAML using only SCC-level edges.
    """

    result = {"scc_dag": []}
    topo_order = nx.topological_sort(scc_dag)

    for scc_id in topo_order:
        members = sorted(scc_id.split(','))

        # incoming edges (with operations)
        incoming_edges = []
        for src_scc, _, data in scc_dag.in_edges(scc_id, data=True):
            incoming_edges.append(f"{src_scc} -> {data['operation']}")

        # outgoing edges (with operations)
        outgoing_edges = []
        for _, dst_scc, data in scc_dag.out_edges(scc_id, data=True):
            outgoing_edges.append(f"{data['operation']} -> {dst_scc}")

        result["scc_dag"].append({
            "id": scc_id,
            "members": members,
            "incoming": sorted(incoming_edges),
            "outgoing": sorted(outgoing_edges),
        })

    return yaml.dump(result, sort_keys=False)

def main():
    csv_path = "dataset/RE3-OB/adservice_f3/1/traces.csv"   # your file name
    #csv_path = "dataset/RE3-TT/ts-auth-service_f1/1/traces.csv"   # your file name
    edges = extract_edges_from_trace(csv_path)

    print("Extracted edges:")
    for e in edges:
        print(e)

    G = build_graph(edges)
    scc_list, scc_map = compute_sccs(G)
    scc_dag = build_scc_dag(G, scc_map)
    visualize_graph_dag(scc_dag)
    print(graph_to_text(scc_dag))
    print(serialize_scc_dag_to_yaml(scc_dag))

if __name__ == '__main__':
    main()
