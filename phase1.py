"""
Phase 1 — Data Collection, Preprocessing, Inverted Index
General-purpose corpus via SerpAPI (any topic, not just medical).
Run once to build corpus.pkl and corpus.xlsx.
"""

import re
import math
import pickle
import os
import time
import requests
import urllib3
import nltk
import pandas as pd
from collections import defaultdict

# ── SSL fix for restricted networks ──────────────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_orig = requests.Session.request
def _no_ssl(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig(self, method, url, **kwargs)
requests.Session.request = _no_ssl

from serpapi import GoogleSearch
import fitz  # PyMuPDF

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


# ---------------------------------------------------
# Config — broad general queries across many fields
# ---------------------------------------------------

QUERIES = [
    # Sports / Football
    "football Champions League 2024",
    "NBA basketball players stats 2024",
    "FIFA World Cup history winners",
    # Food
    "best pasta recipes Italian cuisine",
    "healthy breakfast nutrition diet",
    # Technology
    "artificial intelligence machine learning 2024",
    "Python programming web development",
    # Medical
    "diabetes insulin treatment blood sugar",
    "cancer immunotherapy checkpoint inhibitors",
    "COVID-19 vaccine efficacy booster",
    "Alzheimer disease biomarker diagnosis",
    "hypertension blood pressure management",
    # Science
    "climate change global warming solutions",
    "space exploration NASA Mars mission",
    # Movies / Entertainment
    "best movies 2024 Hollywood blockbuster",
    "Netflix series popular streaming 2024",
    # Business
    "stock market investment strategies 2024",
    "startup entrepreneurship business tips",
    # Travel
    "best travel destinations Europe 2024",
    "Dubai tourism attractions guide",
]

HEADERS   = {"User-Agent": "Mozilla/5.0"}
MAX_RETRY = 3


# ---------------------------------------------------
# Step 1 — Collect via SerpAPI
# ---------------------------------------------------

def safe_search(params, retries=MAX_RETRY):
    for attempt in range(1, retries + 1):
        try:
            result = GoogleSearch(params).get_dict()
            if "error" in result:
                print(f"  SerpAPI error: {result['error']}")
                return {}
            return result
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    print("  All retries failed — skipping.")
    return {}


def collect_corpus(serp_key):
    corpus   = {}
    metadata = {}
    doc_id   = 0

    for query in QUERIES:
        print(f"  [Fetching] {query}")

        params  = {
            "q"      : query,
            "api_key": serp_key,
            "num"    : 10,
            "engine" : "google",
        }
        results = safe_search(params)

        for r in results.get("organic_results", []):
            title   = r.get("title",   "")
            snippet = r.get("snippet", "")
            link    = r.get("link",    "")
            if not title and not snippet:
                continue
            corpus[f"doc_{doc_id}"]   = title + ". " + snippet
            metadata[f"doc_{doc_id}"] = {
                "title": title,
                "link" : link,
                "image": "N/A",
                "type" : "text",
            }
            doc_id += 1

        time.sleep(1.0)

    print(f"\n  Total documents collected: {len(corpus)}")
    return corpus, metadata


# ---------------------------------------------------
# Step 2 — Preprocess
# ---------------------------------------------------

lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

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
# Step 3 — Inverted Index + TF / DF / IDF
# ---------------------------------------------------

def build_index(preprocessed_corpus):
    inverted_index = defaultdict(lambda: defaultdict(int))
    for doc_id, tokens in preprocessed_corpus.items():
        for token in tokens:
            inverted_index[token][doc_id] += 1

    N   = len(preprocessed_corpus)
    df  = {term: len(postings) for term, postings in inverted_index.items()}
    idf = {term: math.log(N / df[term]) for term in df}
    tf  = {
        term: {
            doc_id: freq / len(preprocessed_corpus[doc_id])
            for doc_id, freq in postings.items()
        }
        for term, postings in inverted_index.items()
    }
    return dict(inverted_index), tf, df, idf


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def build_phase1(serp_key=None):
    if serp_key is None:
        SERP_KEY_HARDCODED = ""   # paste key here to skip prompt
        serp_key = SERP_KEY_HARDCODED or input("Enter SerpAPI key: ").strip()

    if not serp_key:
        print("No key provided. Exiting.")
        return None

    print("\n╔══════════════════════════════════════╗")
    print("║   Phase 1 — Building General Corpus  ║")
    print("╚══════════════════════════════════════╝\n")

    corpus, metadata            = collect_corpus(serp_key)
    if not corpus:
        print("No documents collected. Check your key and network.")
        return None

    print("\nPreprocessing documents ...")
    preprocessed_corpus         = {did: preprocess(text) for did, text in corpus.items()}

    print("Building inverted index ...")
    inverted_index, tf, df, idf = build_index(preprocessed_corpus)

    pd.DataFrame({
        "doc_id": list(corpus.keys()),
        "text"  : list(corpus.values()),
        "type"  : [metadata[k]["type"]  for k in corpus],
        "link"  : [metadata[k]["link"]  for k in corpus],
        "title" : [metadata[k]["title"] for k in corpus],
    }).to_excel("corpus.xlsx", index=False)

    data = {
        "corpus"             : corpus,
        "metadata"           : metadata,
        "preprocessed_corpus": preprocessed_corpus,
        "inverted_index"     : inverted_index,
        "tf"                 : tf,
        "df"                 : df,
        "idf"                : idf,
    }
    with open("corpus.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"\n✓ Documents collected   : {len(corpus)}")
    print(f"✓ Unique terms in index : {len(inverted_index)}")
    print(f"✓ Saved: corpus.pkl, corpus.xlsx")
    print(f"\n→ Now run: python app.py  then open http://127.0.0.1:5000\n")
    return data


if __name__ == "__main__":
    build_phase1()
