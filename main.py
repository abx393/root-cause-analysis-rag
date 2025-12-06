from typing import TypedDict, List
from langchain_openai import ChatOpenAI

from graph_builder import (
    extract_edges_from_trace,
    build_graph,
    compute_sccs,
    build_scc_dag,
    serialize_scc_dag_to_yaml,
)
from log_parser import get_stacktrace_from_logs


# -----------------------------
# Structured output (ranked list)
# -----------------------------
class RCARankedResult(TypedDict):
    ranked_services: List[str]


# -----------------------------
# Build LLM prompt
# -----------------------------
def build_llm_message(graph_text: str, stacktrace: str, services: list[str]) -> str:
    service_list = "\n".join(f"- {s}" for s in services)

    return (
        "You are performing Root Cause Analysis for a distributed microservice system.\n"
        "Below is the microservice dependency graph:\n\n"
        f"{graph_text}\n\n"
        "Here is the list of ALL services involved in this trace:\n"
        f"{service_list}\n\n"
        "You MUST return a STRICT ranking of these services from MOST likely root cause "
        "to LEAST likely root cause, based ONLY on:\n"
        " - the graph dependencies\n"
        " - stacktrace content\n\n"
        f"Stacktrace:\n{stacktrace}\n\n"
        "Return ONLY a JSON object with a ranked_services list."
    )


# -----------------------------
# LLM invocation
# -----------------------------
def invoke_llm(message):
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured = llm.with_structured_output(RCARankedResult)
    return structured.invoke(message)


# -----------------------------
# MAIN (one case example)
# -----------------------------
def main():
    traces_path = "dataset/RE3-OB/cartservice_f1/1/traces.csv"
    logs_path = "dataset/RE3-OB/cartservice_f1/1/logs.csv"

    # Build graph
    edges = extract_edges_from_trace(traces_path)
    G = build_graph(edges)
    scc_list, scc_map = compute_sccs(G)
    scc_dag = build_scc_dag(G, scc_map)

    graph_text = serialize_scc_dag_to_yaml(scc_dag)

    # Extract stacktrace
    stacktraces = get_stacktrace_from_logs(logs_path)
    first_stack = list(stacktraces)[0] if len(stacktraces) > 0 else ""

    # Services (nodes in SCC DAG)
    services = list(scc_dag.nodes)

    # Build prompt
    message = build_llm_message(graph_text, first_stack, services)

    # Run LLM
    result = invoke_llm(message)
    print("LLM Ranked Output:", result)


if __name__ == "__main__":
    main()
