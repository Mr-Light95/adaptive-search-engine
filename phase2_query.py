"""
Phase 2 — Query Processing + Query Expansion
Steps:
  4. Query Processing  : preprocess → inverted index → TF-IDF ranking
  5. Query Expansion   : RM3 pseudo-relevance feedback + WordNet synonyms + BERT
"""

import re
import math
import time
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

import nltk
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


# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r"http\S+|www\S+", " ", text)
    text   = re.sub(r"[^a-z\s]",       " ", text)
    text   = re.sub(r"\s+",            " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(lemmatizer.lemmatize(t))
              for t in tokens if t not in stop_words and len(t) > 2]
    return tokens


# ---------------------------------------------------
# Step 4 — TF-IDF Ranking
# ---------------------------------------------------

def tfidf_score(query_tokens, inverted_index, tf, idf, preprocessed_corpus):
    scores = defaultdict(float)
    N      = len(preprocessed_corpus)
    for term in query_tokens:
        if term not in inverted_index:
            continue
        q_idf = idf.get(term, math.log(N / 1))
        for doc_id, freq in inverted_index[term].items():
            doc_tf = tf.get(term, {}).get(doc_id, 0)
            scores[doc_id] += doc_tf * q_idf
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def query_processing(query, data, top_k=10):
    inverted_index      = data["inverted_index"]
    tf                  = data["tf"]
    df                  = data["df"]
    idf                 = data["idf"]
    preprocessed_corpus = data["preprocessed_corpus"]
    metadata            = data["metadata"]

    start        = time.time()
    query_tokens = preprocess(query)
    ranked       = tfidf_score(query_tokens, inverted_index, tf, idf, preprocessed_corpus)
    elapsed      = round(time.time() - start, 4)

    rows = []
    for doc_id, score in ranked[:top_k]:
        meta = metadata.get(doc_id, {})
        rows.append({
            "doc_id": doc_id,
            "score" : round(score, 4),
            "type"  : meta.get("type",  ""),
            "title" : meta.get("title", "")[:80],
            "link"  : meta.get("link",  ""),
        })
    return pd.DataFrame(rows), query_tokens, ranked, elapsed


# ---------------------------------------------------
# Step 5a — WordNet Expansion
# ---------------------------------------------------

def wordnet_expansion(query_tokens, max_syn=2):
    expanded = list(query_tokens)
    for token in query_tokens:
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


# ---------------------------------------------------
# Step 5b — RM3 Pseudo-Relevance Feedback
# ---------------------------------------------------

def rm3_expansion(query_tokens, ranked_docs, preprocessed_corpus,
                  top_docs=3, top_terms=5, lam=0.6):
    term_scores   = defaultdict(float)
    feedback_docs = [doc_id for doc_id, _ in ranked_docs[:top_docs]]

    for doc_id in feedback_docs:
        tokens  = preprocessed_corpus.get(doc_id, [])
        doc_len = len(tokens)
        if doc_len == 0:
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


# ---------------------------------------------------
# Step 5c — BERT Expansion
# ---------------------------------------------------

def bert_expansion(query, ranked_docs, preprocessed_corpus, top_docs=3, top_terms=5):
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return preprocess(query), []

    model        = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb    = model.encode(query, convert_to_tensor=True)
    feedback     = [doc_id for doc_id, _ in ranked_docs[:top_docs]]
    candidates   = list({t for doc_id in feedback
                          for t in preprocessed_corpus.get(doc_id, [])})[:200]
    if not candidates:
        return preprocess(query), []

    cand_embs = model.encode(candidates, convert_to_tensor=True)
    scores    = util.cos_sim(query_emb, cand_embs)[0].cpu().numpy()
    top_idx   = np.argsort(scores)[::-1][:top_terms]
    new_terms = [candidates[i] for i in top_idx]
    q_tokens  = preprocess(query)
    expanded  = q_tokens + [t for t in new_terms if t not in q_tokens]
    return expanded, new_terms


# ---------------------------------------------------
# Full Phase 2 Pipeline
# ---------------------------------------------------

def run_phase2(query, data, expansion="wordnet", top_k=10):
    preprocessed_corpus = data["preprocessed_corpus"]
    inverted_index      = data["inverted_index"]
    tf                  = data["tf"]
    idf                 = data["idf"]

    # Initial retrieval
    results_df, q_tokens, ranked, elapsed = query_processing(query, data, top_k)

    if not ranked:
        return results_df, results_df, [], [], []

    # Expansion
    new_terms = []
    if expansion == "wordnet":
        expanded_tokens = wordnet_expansion(q_tokens)
        new_terms = [t for t in expanded_tokens if t not in q_tokens]
    elif expansion == "rm3":
        expanded_tokens, new_terms = rm3_expansion(q_tokens, ranked, preprocessed_corpus)
    elif expansion == "bert":
        expanded_tokens, new_terms = bert_expansion(query, ranked, preprocessed_corpus)
    else:
        expanded_tokens = q_tokens

    # Re-rank
    ranked_expanded = tfidf_score(expanded_tokens, inverted_index, tf, idf, preprocessed_corpus)

    rows = []
    for doc_id, score in ranked_expanded[:top_k]:
        meta = data["metadata"].get(doc_id, {})
        rows.append({
            "doc_id": doc_id,
            "score" : round(score, 4),
            "type"  : meta.get("type",  ""),
            "title" : meta.get("title", "")[:80],
            "link"  : meta.get("link",  ""),
        })
    expanded_df = pd.DataFrame(rows)
    return results_df, expanded_df, q_tokens, expanded_tokens, new_terms


def load_data(path="corpus.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
