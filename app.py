"""
Adaptive AI Search Engine — Flask Backend
General-purpose: football, food, medical, tech, movies, anything.
Shows all IR calculations: TF-IDF, tokens, expansion terms, metrics.
"""

import os
import sys
import json
import math
import time
import pickle
import re
from collections import defaultdict
from flask import Flask, render_template, request, jsonify

import anthropic
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

app = Flask(__name__)

# ── Load corpus ───────────────────────────────────────────────────
DATA = None

def load_corpus():
    global DATA
    path = os.path.join(os.path.dirname(__file__), "corpus.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            DATA = pickle.load(f)
        print(f"✓ Corpus: {len(DATA['corpus'])} docs | {len(DATA['inverted_index'])} terms")
    else:
        print("⚠  corpus.pkl not found — run phase1.py first")
        DATA = None


# ── IR helpers ────────────────────────────────────────────────────
import nltk, urllib3, requests as req
urllib3.disable_warnings()
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus   import stopwords, wordnet
from nltk.stem     import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]",       " ", text)
    text   = re.sub(r"\s+",            " ", text).strip()
    tokens = word_tokenize(text)
    return [stemmer.stem(lemmatizer.lemmatize(t))
            for t in tokens if t not in stop_words and len(t) > 2]

def tfidf_score(query_tokens, data):
    scores = defaultdict(float)
    N      = len(data["preprocessed_corpus"])
    for term in query_tokens:
        if term not in data["inverted_index"]:
            continue
        q_idf = data["idf"].get(term, math.log(N))
        for doc_id, freq in data["inverted_index"][term].items():
            scores[doc_id] += data["tf"].get(term, {}).get(doc_id, 0) * q_idf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def wordnet_expansion(tokens, max_syn=2):
    expanded = list(tokens)
    for token in tokens:
        count = 0
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                word = stemmer.stem(lemma.name().replace("_", " ").lower())
                if word not in expanded and count < max_syn:
                    expanded.append(word)
                    count += 1
            if count >= max_syn:
                break
    return expanded

def rm3_expansion(query_tokens, ranked, preprocessed_corpus, top_docs=3, top_terms=5):
    term_scores = defaultdict(float)
    for doc_id in [d for d, _ in ranked[:top_docs]]:
        tokens  = preprocessed_corpus.get(doc_id, [])
        doc_len = len(tokens)
        if not doc_len:
            continue
        tf_local = defaultdict(int)
        for t in tokens:
            tf_local[t] += 1
        for term, freq in tf_local.items():
            term_scores[term] += freq / doc_len
    for t in query_tokens:
        term_scores.pop(t, None)
    new_terms = sorted(term_scores, key=lambda x: term_scores[x], reverse=True)[:top_terms]
    return list(query_tokens) + new_terms, new_terms


# ── Metrics ───────────────────────────────────────────────────────
def precision_at_k(retrieved, relevant):
    if not retrieved:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(retrieved)

def recall_at_k(retrieved, relevant):
    if not relevant:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(relevant)

def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0

def ndcg(retrieved, relevant):
    def dcg(lst):
        return sum((1 if d in relevant else 0) / math.log2(i + 2)
                   for i, d in enumerate(lst))
    ideal = sorted(retrieved, key=lambda x: 1 if x in relevant else 0, reverse=True)
    idcg  = dcg(ideal)
    return dcg(retrieved) / idcg if idcg else 0.0

def average_precision(retrieved, relevant):
    hits, score = 0, 0.0
    for i, d in enumerate(retrieved):
        if d in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0


# ── Anthropic — category detection ───────────────────────────────
SYSTEM_PROMPT = """You are a search engine API. Given a query and IR snippets, respond ONLY with valid JSON (no markdown):
{
  "category": "medical|sports|food|technology|movies|travel|science|business|general",
  "tagline": "short creative tagline max 7 words",
  "enriched_results": [
    {
      "id": 1,
      "title": "title here",
      "snippet": "2-sentence relevant snippet",
      "source": "domain.com",
      "score": 0.95,
      "type": "article|video|recipe|match|report|paper",
      "meta": "date or author"
    }
  ]
}
Return 5-8 results. Use the IR snippets provided as your primary source. Score 0-1."""

def call_anthropic(query, ir_snippets):
    ir_ctx = "\n".join(
        f"- [{r['title']}] {r['snippet'][:120]} ({r['source']})"
        for r in ir_snippets
    )
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user",
                   "content": f'Query: "{query}"\n\nIR Results:\n{ir_ctx}'}],
    )
    text  = "".join(b.text for b in msg.content if hasattr(b, "text"))
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    body      = request.get_json(force=True)
    query     = body.get("query", "").strip()
    expansion = body.get("expansion", "none")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if not DATA:
        return jsonify({"error": "corpus.pkl not found. Run phase1.py first."}), 503

    start  = time.time()
    tokens = preprocess(query)

    # ── Baseline TF-IDF ──────────────────────────────────────────
    t0           = time.time()
    ranked       = tfidf_score(tokens, DATA)
    baseline_ms  = round((time.time() - t0) * 1000, 2)

    # ── Expansion ─────────────────────────────────────────────────
    new_terms       = []
    expanded_tokens = tokens

    if expansion == "wordnet" and ranked:
        expanded_tokens = wordnet_expansion(tokens)
        new_terms       = [t for t in expanded_tokens if t not in tokens]
        ranked          = tfidf_score(expanded_tokens, DATA)

    elif expansion == "rm3" and ranked:
        expanded_tokens, new_terms = rm3_expansion(tokens, ranked,
                                                   DATA["preprocessed_corpus"])
        ranked = tfidf_score(expanded_tokens, DATA)

    total_ms = round((time.time() - start) * 1000, 2)

    # ── Top results for Anthropic ─────────────────────────────────
    top10       = ranked[:10]
    ir_snippets = []
    doc_scores  = {}

    for doc_id, score in top10:
        meta  = DATA["metadata"].get(doc_id, {})
        text  = DATA["corpus"].get(doc_id, "")
        doc_scores[doc_id] = round(score, 4)
        ir_snippets.append({
            "title"  : meta.get("title", doc_id)[:80],
            "snippet": text[:150],
            "source" : meta.get("link", "")[:60],
            "score"  : round(score, 4),
            "type"   : meta.get("type", "article"),
        })

    # ── Metrics (pseudo ground truth = top 5 baseline) ───────────
    pseudo_relevant = [d for d, _ in tfidf_score(tokens, DATA)[:5]]
    retrieved_ids   = [d for d, _ in top10]

    p  = round(precision_at_k(retrieved_ids, pseudo_relevant), 3)
    r  = round(recall_at_k(retrieved_ids, pseudo_relevant), 3)
    f  = round(f1(p, r), 3)
    nd = round(ndcg(retrieved_ids, pseudo_relevant), 3)
    ap = round(average_precision(retrieved_ids, pseudo_relevant), 3)

    # ── Per-term IDF values ───────────────────────────────────────
    term_details = []
    for term in tokens[:8]:
        term_details.append({
            "term": term,
            "df"  : DATA["df"].get(term, 0),
            "idf" : round(DATA["idf"].get(term, 0), 4),
            "in_index": term in DATA["inverted_index"],
        })

    # ── Anthropic enrichment ──────────────────────────────────────
    category, tagline, enriched = "general", "", ir_snippets[:8]
    try:
        ai = call_anthropic(query, ir_snippets)
        category = ai.get("category", "general")
        tagline  = ai.get("tagline",  "")
        enriched = ai.get("enriched_results", ir_snippets[:8])
    except Exception as e:
        print(f"Anthropic error: {e}")

    return jsonify({
        # Results
        "category"       : category,
        "tagline"        : tagline,
        "results"        : enriched,
        # IR calculations
        "query_tokens"   : tokens,
        "expanded_tokens": expanded_tokens,
        "new_terms"      : new_terms,
        "expansion_mode" : expansion,
        "term_details"   : term_details,
        "docs_retrieved" : len(ranked),
        # Metrics
        "metrics": {
            "precision"   : p,
            "recall"      : r,
            "f1"          : f,
            "ndcg"        : nd,
            "ap"          : ap,
            "baseline_ms" : baseline_ms,
            "total_ms"    : total_ms,
        },
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    if not DATA:
        return jsonify({"error": "corpus.pkl not found"}), 400
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from evaluate import evaluate as run_eval
        df_b, df_e, map_b, map_e = run_eval(DATA)
        return jsonify({
            "map_baseline": map_b,
            "map_expanded": map_e,
            "baseline"    : df_b.to_dict(orient="records"),
            "expanded"    : df_e.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_corpus()
    app.run(debug=True, port=5000)
