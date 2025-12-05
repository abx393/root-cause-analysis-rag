from typing import TypedDict
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
# Structured output
# -----------------------------
class RCAResult(TypedDict):
    root_cause_service: str


# -----------------------------
# Build LLM prompt
# -----------------------------
def build_llm_message(graph_text: str, stacktrace: str) -> str:
    return (
        "You are performing Root Cause Analysis for a distributed microservice system.\n"
        "Below is the microservice dependency graph:\n\n"
        f"{graph_text}\n\n"
        "Using ONLY this graph and the following stacktrace, identify the single service "
        "most likely responsible for the failure.\n\n"
        f"Stacktrace:\n{stacktrace}\n\n"
        "Return ONLY the service name."
    )


# -----------------------------
# LLM invocation wrapper
# -----------------------------
def invoke_llm(message):
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured = llm.with_structured_output(RCAResult)
    return structured.invoke(message)


# -----------------------------
# MAIN — runs ONE example case
# -----------------------------
def main():
    # One example test case
    traces_path = "dataset/RE3-OB/cartservice_f1/1/traces.csv"
    logs_path   = "dataset/RE3-OB/cartservice_f1/1/logs.csv"

    # Build graph
    edges = extract_edges_from_trace(traces_path)
    G = build_graph(edges)
    scc_list, scc_map = compute_sccs(G)
    scc_dag = build_scc_dag(G, scc_map)

    # Serialize to YAML
    graph_text = serialize_scc_dag_to_yaml(scc_dag)

    # Extract raw stacktrace text
    stacktraces = get_stacktrace_from_logs(logs_path)
    first_stack = list(stacktraces)[0] if len(stacktraces) > 0 else ""

    # Build prompt
    message = build_llm_message(graph_text, first_stack)

    # Run LLM
    result = invoke_llm(message)
    print("LLM Prediction:", result)


if __name__ == "__main__":
    main()
