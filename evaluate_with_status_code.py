import os
import csv
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
from trace_parser import get_error_code_from_trace


# -----------------------------
# Ranked structured output
# -----------------------------
class RCARankedResult(TypedDict):
    ranked_services: List[str]


# -----------------------------
# Build prompt
# -----------------------------
def build_llm_message(graph_text: str, stacktrace: str, services: List[str], non_zero_status_trace) -> str:
    service_list = "\n".join(f"- {s}" for s in services)

    return f"""
        You are performing Root Cause Analysis for a distributed microservice system.
        Below is the microservice dependency graph:
        {graph_text}

        Here is the list of ALL services involved in this trace:
        {service_list}
        
        Here are the details of the trace with the most common non-zero status code among all traces:
        service: {non_zero_status_trace['serviceName']}, method: {non_zero_status_trace['methodName']}, operation: {non_zero_status_trace['operationName']}, duration: {non_zero_status_trace['duration']}, status code: {non_zero_status_trace['statusCode']}
        The destination service of this operation may have an issue.

        Rank ALL services from MOST likely root cause to LEAST likely.
        Return ONLY JSON with field ranked_services.
        
        Stacktrace:
        {stacktrace}
    """

# -----------------------------
# LLM call
# -----------------------------
def invoke_llm(message):
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured = llm.with_structured_output(RCARankedResult)
    return structured.invoke(message)


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate():
    DATASET = "dataset/RE3-OB"

    dataset_name = os.path.basename(DATASET.rstrip("/"))
    OUTPUT_ROOT = os.path.join("results", dataset_name)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    OUTPUT_RESULTS_PATH = os.path.join(OUTPUT_ROOT, "rca_outputs.csv")
    OUTPUT_METRICS_PATH = os.path.join(OUTPUT_ROOT, "rca_metrics.csv")

    results = []  # [folder, case, gt, ranked_list]
    top1_results = []  # [gt, pred]

    for folder in os.listdir(DATASET):
        folder_path = os.path.join(DATASET, folder)
        if not os.path.isdir(folder_path):
            continue

        ground_truth = folder.split("_")[0]

        for case in os.listdir(folder_path):
            case_path = os.path.join(folder_path, case)

            traces_path = os.path.join(case_path, "traces.csv")
            logs_path = os.path.join(case_path, "logs.csv")

            if not (os.path.exists(traces_path) and os.path.exists(logs_path)):
                continue

            # ---- Build graph ----
            edges = extract_edges_from_trace(traces_path)
            G = build_graph(edges)
            scc_list, scc_map = compute_sccs(G)
            scc_dag = build_scc_dag(G, scc_map)

            graph_text = serialize_scc_dag_to_yaml(scc_dag)
            services = list(scc_dag.nodes)

            # ---- Stacktrace ----
            stack = get_stacktrace_from_logs(logs_path)
            stack = list(stack)[0] if len(stack) > 0 else ""

            # Status Code
            error_rows, status_code_counts = get_error_code_from_trace(traces_path)
            codes = list(status_code_counts.keys())
            for code in codes:
                print(f"count of code {code} is {status_code_counts[code]}")

            idx = 0
            while idx < len(error_rows):
                if error_rows[idx]['statusCode'] == codes[1]:
                    break
                idx += 1
            print(error_rows[idx])

            # ---- LLM call ----
            msg = build_llm_message(graph_text, stack, services, error_rows[idx])
            ranked = invoke_llm(msg)["ranked_services"]

            pred_top1 = ranked[0] if ranked else "none"
            correct = 1 if pred_top1.lower() == ground_truth.lower() else 0

            print(f"[{folder}/{case}] GT={ground_truth}, Pred@1={pred_top1}, correct={correct}")

            results.append([folder, case, ground_truth, ranked])
            top1_results.append([ground_truth, pred_top1, correct])

    # ---------------------------------------------------
    # Write CSV
    # ---------------------------------------------------
    with open(OUTPUT_RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "case", "ground_truth", "ranked_services"])
        for folder, case, gt, ranked in results:
            writer.writerow([folder, case, gt, ";".join(ranked)])

    # ---------------------------------------------------
    # BASIC METRICS (Top-1 Accuracy + Per-Service)
    # ---------------------------------------------------
    total = len(top1_results)
    correct_total = sum(r[2] for r in top1_results)
    accuracy = correct_total / total if total else 0

    print("\n================ METRICS SUMMARY ================\n")
    print(f"Total Cases: {total}")
    print(f"Correct Predictions: {correct_total}")
    print(f"Overall Accuracy (Top-1): {accuracy:.3f}")

    print("\nPer-service accuracy:")
    service_stats = {}
    for gt, pred, correct in top1_results:
        if gt not in service_stats:
            service_stats[gt] = {"total": 0, "correct": 0}
        service_stats[gt]["total"] += 1
        service_stats[gt]["correct"] += correct

    for svc, stats in service_stats.items():
        svc_acc = stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {svc}: {stats['correct']}/{stats['total']} = {svc_acc:.3f}")

    print("\n=================================================\n")

    # ---------------------------------------------------
    # RCAEval METRICS: AC@k and AVG@5
    # ---------------------------------------------------
    ac = [0 for i in range(6)]

    for _, _, gt, ranked in results:
        for i in range(1, 6):
            if gt in ranked[:i]: ac[i] += 1

    avg5 = 0
    for i in range(1, 6):
        ac[i] /= total
        avg5 += ac[i]
    avg5 /= 5

    with open(OUTPUT_METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        # Global metrics
        writer.writerow(["metric", "value"])
        writer.writerow(["total_cases", total])
        writer.writerow(["top1_accuracy", accuracy])
        writer.writerow(["AC@1", ac[1]])
        writer.writerow(["AC@3", ac[3]])
        writer.writerow(["AC@5", ac[5]])
        writer.writerow(["Avg@5", avg5])

        # Per-service accuracy
        writer.writerow([])
        writer.writerow(["service", "correct", "total", "accuracy"])

        for svc, stats in service_stats.items():
            svc_acc = stats["correct"] / stats["total"] if stats["total"] else 0
            writer.writerow([svc, stats["correct"], stats["total"], svc_acc])

    print("\n================ RCAEval METRICS ================\n")
    print(f"Total Cases: {total}")
    print(f"AC@1: {ac[1]:.3f}")
    print(f"AC@3: {ac[3]:.3f}")
    print(f"AC@5: {ac[5]:.3f}")
    print(f"Avg@5 (lower better): {avg5:.3f}")
    print("\n=================================================\n")

    print("Evaluation complete!")
    print(f"Predictions written to: {OUTPUT_RESULTS_PATH}")
    print(f"Metrics written to:     {OUTPUT_METRICS_PATH}")


if __name__ == "__main__":
    evaluate()
