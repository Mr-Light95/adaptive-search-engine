"""
Phase 2 — Step 7: Evaluation
Metrics: Precision, Recall, F1, Fbeta, AP, MAP, nDCG
"""

import math
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase2_query import preprocess, tfidf_score, rm3_expansion, load_data


# ---------------------------------------------------
# Metric Functions
# ---------------------------------------------------

def precision(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)

def recall(retrieved, relevant):
    retrieved, relevant = set(retrieved), set(relevant)
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)

def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0

def fbeta_score(p, r, beta=2):
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r) if (p + r) else 0.0

def average_precision(retrieved, relevant):
    score, hits = 0.0, 0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0

def dcg(retrieved, relevant):
    return sum(
        (1 if doc in relevant else 0) / math.log2(i + 2)
        for i, doc in enumerate(retrieved)
    )

def ndcg(retrieved, relevant):
    ideal = sorted(retrieved, key=lambda x: 1 if x in relevant else 0, reverse=True)
    idcg  = dcg(ideal, relevant)
    return dcg(retrieved, relevant) / idcg if idcg else 0.0


# ---------------------------------------------------
# Build pseudo ground truth
# ---------------------------------------------------

def build_ground_truth(queries, data, top_relevant=5):
    ground_truth = {}
    for query in queries:
        tokens  = preprocess(query)
        ranked  = tfidf_score(tokens, data["inverted_index"],
                              data["tf"], data["idf"], data["preprocessed_corpus"])
        ground_truth[query] = [doc_id for doc_id, _ in ranked[:top_relevant]]
    return ground_truth


# ---------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------

def evaluate(data, top_k=10):
    test_queries = [
        "football Champions League",
        "diabetes insulin treatment",
        "cancer immunotherapy",
        "COVID-19 vaccine",
        "artificial intelligence machine learning",
        "best pasta recipe food",
        "hypertension blood pressure",
        "mental health depression",
        "space exploration NASA",
        "stock market investment",
    ]

    ground_truth     = build_ground_truth(test_queries, data, top_relevant=5)
    results_baseline = []
    results_expanded = []
    ap_baseline      = []
    ap_expanded      = []

    for i, query in enumerate(test_queries):
        q_label  = f"Q{i+1}"
        relevant = ground_truth[query]
        tokens   = preprocess(query)

        # Baseline
        start_b  = time.time()
        ranked_b = tfidf_score(tokens, data["inverted_index"],
                               data["tf"], data["idf"], data["preprocessed_corpus"])
        time_b   = round(time.time() - start_b, 5)
        ret_b    = [d for d, _ in ranked_b[:top_k]]

        p_b  = precision(ret_b, relevant)
        r_b  = recall(ret_b, relevant)
        f1_b = f1_score(p_b, r_b)
        fb_b = fbeta_score(p_b, r_b, beta=2)
        ap_b = average_precision(ret_b, relevant)
        nd_b = ndcg(ret_b, relevant)
        ap_baseline.append(ap_b)
        results_baseline.append([q_label, query[:30], round(p_b,3), round(r_b,3),
                                  round(f1_b,3), round(fb_b,3), round(ap_b,3),
                                  round(nd_b,3), time_b])

        # RM3 Expanded
        start_e         = time.time()
        exp_tokens, _   = rm3_expansion(tokens, ranked_b, data["preprocessed_corpus"])
        ranked_e        = tfidf_score(exp_tokens, data["inverted_index"],
                                      data["tf"], data["idf"], data["preprocessed_corpus"])
        time_e          = round(time.time() - start_e, 5)
        ret_e           = [d for d, _ in ranked_e[:top_k]]

        p_e  = precision(ret_e, relevant)
        r_e  = recall(ret_e, relevant)
        f1_e = f1_score(p_e, r_e)
        fb_e = fbeta_score(p_e, r_e, beta=2)
        ap_e = average_precision(ret_e, relevant)
        nd_e = ndcg(ret_e, relevant)
        ap_expanded.append(ap_e)
        results_expanded.append([q_label, query[:30], round(p_e,3), round(r_e,3),
                                  round(f1_e,3), round(fb_e,3), round(ap_e,3),
                                  round(nd_e,3), time_e])

    cols     = ["Query","Text","Precision","Recall","F1","Fbeta","AP","nDCG","Time(s)"]
    df_base  = pd.DataFrame(results_baseline, columns=cols)
    df_exp   = pd.DataFrame(results_expanded,  columns=cols)
    MAP_base = round(float(np.mean(ap_baseline)), 4)
    MAP_exp  = round(float(np.mean(ap_expanded)),  4)

    _plot(df_base, df_exp, MAP_base, MAP_exp)
    return df_base, df_exp, MAP_base, MAP_exp


# ---------------------------------------------------
# Plot
# ---------------------------------------------------

def _plot(df_base, df_exp, MAP_base, MAP_exp):
    metrics = ["Precision", "Recall", "F1", "nDCG"]
    x       = np.arange(len(df_base))
    w       = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes      = axes.flatten()
    for i, m in enumerate(metrics):
        axes[i].bar(x - w/2, df_base[m], w, label="Baseline",  color="#3b82f6")
        axes[i].bar(x + w/2, df_exp[m],  w, label="RM3 Expanded", color="#10b981")
        axes[i].set_title(m, fontweight="bold")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(df_base["Query"])
        axes[i].set_ylim(0, 1.15)
        axes[i].legend()
        axes[i].grid(axis="y", alpha=0.3)
    fig.suptitle(f"Baseline MAP={MAP_base}  vs  RM3 Expanded MAP={MAP_exp}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    data = load_data()
    df_b, df_e, m_b, m_e = evaluate(data)
    print("\n--- Baseline ---")
    print(df_b.to_string(index=False))
    print(f"\nMAP Baseline : {m_b}")
    print("\n--- RM3 Expanded ---")
    print(df_e.to_string(index=False))
    print(f"\nMAP Expanded : {m_e}")
