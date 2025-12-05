import os
import csv
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
# Structured LLM output
# -----------------------------
class RCAResult(TypedDict):
    root_cause_service: str


# -----------------------------
# LLM Prompt Builder (FAST)
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
# Fast LLM call
# -----------------------------
def invoke_llm(message):
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured = llm.with_structured_output(RCAResult)
    return structured.invoke(message)


# -----------------------------
# MAIN EVALUATION LOOP (FAST)
# -----------------------------
def evaluate():
    DATASET = "dataset/RE3-OB"

    results = []

    for folder in os.listdir(DATASET):
        folder_path = os.path.join(DATASET, folder)
        if not os.path.isdir(folder_path):
            continue

        ground_truth = folder.split("_")[0]

        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)

            traces_path = os.path.join(case_path, "traces.csv")
            logs_path   = os.path.join(case_path, "logs.csv")

            if not (os.path.exists(traces_path) and os.path.exists(logs_path)):
                continue

            # ---- Build graph ----
            edges = extract_edges_from_trace(traces_path)
            G = build_graph(edges)
            scc_list, scc_map = compute_sccs(G)
            scc_dag = build_scc_dag(G, scc_map)

            graph_text = serialize_scc_dag_to_yaml(scc_dag)

            # ---- FAST stacktrace extraction ----
            stack = get_stacktrace_from_logs(logs_path)
            stack = list(stack)[0] if len(stack) > 0 else ""

            # ---- Build prompt ----
            msg = build_llm_message(graph_text, stack)

            # ---- LLM prediction ----
            pred = invoke_llm(msg)["root_cause_service"]

            correct = 1 if pred.lower() == ground_truth.lower() else 0

            print(f"[{folder}/{case}] GT={ground_truth}, Pred={pred}, correct={correct}")

            results.append([folder, case, ground_truth, pred, correct])

    # -----------------------------
    # Write CSV
    # -----------------------------
    out_path = "rca_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "case", "ground_truth", "prediction", "correct"])
        writer.writerows(results)

    # -----------------------------
    # SUMMARY METRICS
    # -----------------------------
    total = len(results)
    correct_total = sum(r[4] for r in results)
    accuracy = correct_total / total if total else 0

    print("\n================ METRICS SUMMARY ================\n")
    print(f"Total Cases: {total}")
    print(f"Correct Predictions: {correct_total}")
    print(f"Overall Accuracy: {accuracy:.3f}")

    print("\nPer-service accuracy:")
    service_stats = {}
    for folder, case, gt, pred, correct in results:
        if gt not in service_stats:
            service_stats[gt] = {"total": 0, "correct": 0}
        service_stats[gt]["total"] += 1
        service_stats[gt]["correct"] += correct

    for svc, stats in service_stats.items():
        svc_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {svc}: {stats['correct']}/{stats['total']} = {svc_acc:.3f}")

    print("\n=================================================\n")
    print(f"Evaluation complete! Results written to {out_path}")


if __name__ == "__main__":
    evaluate()
