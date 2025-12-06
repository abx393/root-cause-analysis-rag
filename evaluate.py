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


# -----------------------------
# Ranked structured output
# -----------------------------
class RCARankedResult(TypedDict):
    ranked_services: List[str]


# -----------------------------
# Build prompt
# -----------------------------
from typing import List

def build_llm_message(graph_text: str, stacktrace: str, services: List[str]) -> str:
    service_list = "\n".join(f"- {s}" for s in services)

    return (
        "You are performing Root Cause Analysis for a distributed microservice system.\n"
        "Below is the microservice dependency graph:\n\n"
        f"{graph_text}\n\n"
        "Here is the list of ALL services involved in this trace:\n"
        f"{service_list}\n\n"
        "Rank ALL services from MOST likely root cause to LEAST likely.\n"
        "Return ONLY JSON with field ranked_services.\n\n"
        f"Stacktrace:\n{stacktrace}\n"
    )


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

    results = []  # each entry: [folder, case, gt, ranked_list]

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

            # Build graph
            edges = extract_edges_from_trace(traces_path)
            G = build_graph(edges)
            scc_list, scc_map = compute_sccs(G)
            scc_dag = build_scc_dag(G, scc_map)

            graph_text = serialize_scc_dag_to_yaml(scc_dag)
            services = list(scc_dag.nodes)

            # Stacktrace
            stack = get_stacktrace_from_logs(logs_path)
            stack = list(stack)[0] if len(stack) > 0 else ""

            # LLM call
            msg = build_llm_message(graph_text, stack, services)
            ranked = invoke_llm(msg)["ranked_services"]

            print(f"[{folder}/{case}] -> Ranked[0] = {ranked[0]} (GT={ground_truth})")

            results.append([folder, case, ground_truth, ranked])

    # ---------------------------------------------------
    # Write expanded CSV
    # ---------------------------------------------------
    with open("rca_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "case", "ground_truth", "ranked_services"])
        for r in results:
            writer.writerow(r)

    # ---------------------------------------------------
    # METRICS: AC@1, AC@3, AC@5, AVG@5
    # ---------------------------------------------------
    total = len(results)
    ac1 = ac3 = ac5 = 0
    avg5 = 0

    for _, _, gt, ranked in results:
        if gt in ranked[:1]: ac1 += 1
        if gt in ranked[:3]: ac3 += 1
        if gt in ranked[:5]: ac5 += 1

        # Avg rank @5 (lower = better)
        if gt in ranked[:5]:
            avg5 += ranked.index(gt) + 1
        else:
            avg5 += 6  # penalize if not in top 5

    # ================= OLD METRICS (Basic Summary) =================
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


# ================= NEW METRICS (RCAEval AC@k / AVG@k) =================
print("\n================ RCAEval METRICS ================\n")
print(f"Total Cases: {total}")
print(f"AC@1: {ac1/total:.3f}")
print(f"AC@3: {ac3/total:.3f}")
print(f"AC@5: {ac5/total:.3f}")
print(f"Avg@5 (lower better): {avg5/total:.3f}")
print("\n=================================================\n")



if __name__ == "__main__":
    evaluate()
