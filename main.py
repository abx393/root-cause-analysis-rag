from enum import Enum
import networkx as nx

from langchain_openai import ChatOpenAI

from typing import TypedDict, Literal

from graph_builder import build_graph, build_scc_dag, compute_sccs, extract_edges_from_trace
from log_parser import get_stacktrace_from_logs



# def invoke_llm(message):
#     llm = ChatOpenAI(model="gpt-4o-mini")
#     structured_llm = llm.with_structured_output(RCAResult)

#     result = structured_llm.invoke(message)
#     print(result)

def main():
    traces_path = "dataset/RE3-OB/cartservice_f1/1/traces.csv"   # your file name
    edges = extract_edges_from_trace(traces_path)

    print("Extracted edges:")
    for e in edges:
        print(e)

    G = build_graph(edges)
    scc_list, scc_map = compute_sccs(G)
    scc_dag = build_scc_dag(G, scc_map)

    ServiceType = Enum("ServiceType", {node : node for node in scc_dag.nodes})

    class RCAResult(TypedDict):
        root_cause_service: ServiceType

    log_path = "dataset/RE3-OB/cartservice_f1/1/logs.csv"
    stacktraces = get_stacktrace_from_logs(log_path)
    first_stacktrace = list(stacktraces)[0]

    msg = f"Identify the root cause of the bug given the following stacktrace: {first_stacktrace}"
    # invoke_llm(msg)
    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.with_structured_output(RCAResult).invoke(msg)
    print(result)

if __name__ == "__main__":
    main()
