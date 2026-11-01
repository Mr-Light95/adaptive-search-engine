# Adaptive Search Engine

An educational information retrieval project with a Flask interface, TF-IDF document ranking and query expansion.

## Source features

- Collect search snippets using SerpAPI and build an inverted index with TF, DF and IDF values.
- Preprocess English text with tokenization, stopword removal, lemmatization and stemming.
- Compare baseline search with WordNet expansion and pseudo-relevance feedback (named RM3 in the source).
- Show query tokens, term statistics, retrieved results and timing in a web interface.
- Optional Anthropic API result enrichment; the separate query module also includes a sentence-transformer expansion function.
- Evaluation code for precision, recall, F1, average precision, MAP and nDCG.

## Local setup

```bash
pip install -r requirements.txt
python phase1.py
python app.py
```

Run commands from the repository root. `phase1.py` prompts for a SerpAPI key and generates `corpus.pkl` and `corpus.xlsx`. API usage may incur charges. To use the optional enrichment, set your own `ANTHROPIC_API_KEY` environment variable before starting the app. Open `http://127.0.0.1:5000`.

## Evaluation and scope

Evaluation uses **pseudo relevance labels derived from the baseline retrieval**, rather than independent human relevance judgments. Its numbers are illustrative comparisons, not evidence of validated retrieval accuracy. No benchmark improvement is claimed.

The collected corpus and API credentials are not included. Runtime requires the listed packages, NLTK resources and any external services you choose to use. The source collection script disables HTTPS certificate verification; restore normal verification before using it outside a controlled local coursework environment. The expansion implementation is a simplified feedback approach rather than a validated full RM3 model. End-to-end execution with external APIs was not verified for this packaged snapshot.

## Files

- `phase1.py`: corpus collection, preprocessing and indexing
- `phase2_query.py`: ranking and expansion functions
- `app.py`: Flask routes and optional enrichment
- `evaluate.py`: illustrative evaluation and plots
- `templates/index.html`: search interface
