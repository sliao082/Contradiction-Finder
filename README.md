# Contradiction Finder for Research Reading

Contradiction Finder for Research Reading is a Streamlit web application that helps students, researchers, and professionals inspect possible contradictions across scientific paper abstracts. Given a research topic, the app retrieves papers from Semantic Scholar, extracts claim-like statements, pairs related claims across papers, scores possible contradictions with a local Hugging Face NLI model, and uses Groq for structured claim validation/recovery and cautious conflict-card explanations.

The output is intentionally framed as **possible contradictions**. It is a literature-review aid, not a scientific fact checker: every card should be read as a lead for closer reading, not as proof that one paper is wrong.

## What the App Does

1. Accepts a literature-review topic such as `retrieval augmented generation hallucination reduction`.
2. Retrieves paper titles, abstracts, authors, years, venues, citation counts, and URLs from Semantic Scholar.
3. Cleans abstracts and splits them into traceable sentences.
4. Extracts and normalizes scientific claims from abstract sentences.
5. Uses SentenceTransformer embeddings to find semantically related cross-paper claim pairs.
6. Uses a local NLI model to score entailment, contradiction, and neutral relationships.
7. Generates cautious conflict cards with side-by-side claims, source evidence, paper links, contradiction scores, reason categories, and explanations.
8. Saves all intermediate artifacts as JSON/JSONL files so the run can be inspected or evaluated later.

## Features

- Research topic search through the Semantic Scholar Academic Graph API.
- Local JSONL cache for retrieved paper metadata and abstracts.
- Abstract cleaning and sentence splitting with traceable sentence IDs.
- Heuristic candidate claim extraction.
- Groq structured JSON validation and normalization for candidate claims.
- Optional Groq abstract-level recovery of up to 3 key claims per paper.
- SentenceTransformer embeddings for normalized claims.
- Cross-paper cosine-similarity candidate pair generation.
- Local Hugging Face NLI scoring in both directions for each pair.
- Groq-generated cautious explanations and reason categories.
- Streamlit tables, metrics, conflict cards, and download buttons.
- Query-specific JSONL outputs for papers, sentences, claims, pairs, scores, cards, and stats.
- Manual-labeling CSV export for reviewing generated conflict cards.
- SciFact evaluation scripts for benchmarking the local NLI scorer.

## Setup

Use Python 3.10 or newer. The commands below are written for PowerShell on Windows because that is the project development environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then fill in:

```text
GROQ_API_KEY=your_groq_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
```

Recommended `.env` values:

```text
GROQ_API_KEY=your_groq_api_key_here
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
NLI_MODEL=cross-encoder/nli-deberta-v3-small
```

The Semantic Scholar key is recommended but not strictly required. The app will attempt unauthenticated access if the key is missing, although rate limits may be stricter. Groq is used for LLM claim validation/recovery and conflict explanations; if `GROQ_API_KEY` is missing, the app falls back to heuristic-only claims and fallback explanations.

The first run may take longer because the SentenceTransformer and NLI models need to download from Hugging Face.

## Run

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

## How to Use the Interface

1. In the sidebar, enter a research topic.
2. Choose the paper limit. The default is 30, and the app caps the value at 50 for MVP performance.
3. Adjust the similarity threshold if needed. Lower values create more candidate pairs; higher values create fewer, closer pairs.
4. Adjust the contradiction threshold if needed. Higher values reduce the number of displayed cards but may miss weaker tensions.
5. Choose the maximum number of conflict cards to display.
6. Keep `Use cached paper retrieval` enabled when repeating a query, so the app can reuse saved Semantic Scholar results.
7. Keep `Use Groq` enabled if you have a Groq key and want LLM claim normalization, claim recovery, and explanations.
8. Keep `Recover up to 3 key claims per abstract` enabled for better recall.
9. Click `Run pipeline`.

After the run completes, use the segmented view control:

- `Conflict cards`: side-by-side possible contradictions with claims, evidence snippets, paper links, NLI scores, reason categories, and cautious explanations.
- `Retrieved papers`: paper metadata table with optional abstract display.
- `Extracted claims`: normalized claims, source paper, claim type, confidence, extraction method, and original sentence.
- `Candidate pairs`: paired claims, semantic similarity, NLI contradiction score, NLI label, source papers, and evidence.
- `Exports`: download JSON/JSONL artifacts and a manual-labeling CSV.

## Recommended Demo Settings

Use the default settings for a first run:

```text
Research topic: retrieval augmented generation hallucination reduction
Paper limit: 30
Similarity threshold: 0.45
Contradiction threshold: 0.65
Max cards: 20
Use cached paper retrieval: enabled
Use Groq: enabled
Recover up to 3 key claims per abstract: enabled
```

If the app produces too many weak cards, raise the contradiction threshold toward `0.80` or `0.90`. If the app produces too few candidate pairs, lower the similarity threshold slightly.

## SciFact Evaluation

The evaluation layer uses Hugging Face `allenai/scifact` as a controlled benchmark for the local NLI contradiction scorer. SciFact is not the same task as open-ended contradiction discovery, but it provides expert-written scientific claims, evidence-containing abstracts, labels, and rationale sentence indices.

Run a validation-set evaluation:

```powershell
python scripts/evaluate_scifact.py --split validation --max-examples 200
```

This writes evaluation artifacts to `data/evaluation/scifact/validation/`. The generated `data/` outputs are intentionally ignored by git.

Useful options:

```powershell
python scripts/evaluate_scifact.py --split validation --max-examples 0
python scripts/evaluate_scifact.py --split validation --thresholds "0.50,0.60,0.70,0.80"
python scripts/evaluate_scifact.py --split validation --exclude-neutral
python scripts/evaluate_scifact.py --split validation --no-charts
```

SciFact evaluation outputs are saved under:

```text
data/evaluation/scifact/{split}/
```

Files saved:

- `scifact_pairs.jsonl`: processed claim/evidence pairs with document IDs, rationale sentence indices, and extracted evidence text.
- `nli_predictions.jsonl`: local NLI labels and scores.
- `metrics.json`: accuracy, macro/weighted F1, grounding metrics, and best threshold by F1.
- `classification_report.txt`: readable sklearn classification report.
- `threshold_sweep.csv`: contradiction threshold, precision, recall, F1, and predicted contradiction count.
- `threshold_sweep.png` and `confusion_matrix.png`: optional charts when matplotlib/seaborn are installed.

Grounding metrics check whether pairs are built from the intended SciFact evidence document and rationale sentence indices. Neutral examples may use an abstract fallback because SciFact does not always provide rationale sentences for not-enough-information cases.

## Manual Evaluation

The app's generated conflict cards can be exported for manual labeling. In Streamlit, use the `Exports` view and download the manual labeling CSV. You can also create it from a saved run:

```powershell
python scripts/export_manual_labels.py data/processed/runs/{query_hash}
```

Fill the `manual_label` column with one of:

- `true_contradiction`
- `partial_or_contextual_contradiction`
- `not_contradiction`
- `unclear`

Then compute manual metrics:

```powershell
python scripts/evaluate_manual_labels.py data/evaluation/manual/{query_hash}_manual_labels.csv
```

Manual metrics estimate precision or acceptance rate for generated conflict cards. They do not estimate recall because the CSV contains only system-predicted conflicts.

## Development Checks

Use these commands before committing changes:

```powershell
python -m compileall app.py src scripts
python scripts/evaluate_scifact.py --split validation --max-examples 2 --output-dir data/evaluation/scifact/smoke --no-charts
```

The SciFact smoke command may need internet access the first time because it downloads the benchmark data and local NLI model dependencies.

## Example Queries

- `retrieval augmented generation hallucination reduction`
- `large language model chain of thought reasoning improves accuracy`
- `exercise intervention depression randomized controlled trial`
- `graph neural networks molecular property prediction`
- `contrastive learning medical image classification`

## Output Files

Each run writes a query-specific folder under:

```text
data/processed/runs/{query_hash}/
```

Files saved per run:

- `papers.jsonl`: retrieved Semantic Scholar paper records.
- `sentences.jsonl`: cleaned abstract sentences with paper IDs.
- `claims.jsonl`: normalized claims with extraction method and confidence.
- `pairs.jsonl`: cross-paper candidate claim pairs with cosine similarity.
- `scores.jsonl`: local NLI entailment, contradiction, and neutral scores.
- `cards.jsonl`: display-ready possible contradiction cards.
- `stats.json`: settings, counts, output folder, and warnings.

Semantic Scholar retrieval caches are stored under:

```text
data/raw/semantic_scholar/
```

Evaluation outputs are stored under:

```text
data/evaluation/
```

These generated data folders are ignored by git so the repository stays lightweight.

## Limitations

- The MVP uses abstracts only and does not download or parse full PDFs.
- NLI scores can produce false positives and false negatives.
- Groq explanations are generated from provided claims and evidence snippets only.
- API rate limits, missing abstracts, or model download issues can reduce output quality.
- A conflict card should be read as a prompt for closer reading, not as proof of a true contradiction.

## Troubleshooting

- If Semantic Scholar retrieval fails, add `SEMANTIC_SCHOLAR_API_KEY` to `.env`, wait for any rate limits to reset, or rerun with cached retrieval enabled.
- If Groq explanations fall back to generic text, check that `GROQ_API_KEY` is set and that the selected `GROQ_MODEL` supports JSON object responses.
- If local NLI scoring fails, reinstall requirements and rerun so Hugging Face models can download.
- If no conflict cards appear, lower the similarity threshold, lower the contradiction threshold, increase the paper limit, or try a broader query.
- If results look overconfident, inspect the `Candidate pairs` view. Many scientific tensions come from different datasets, methods, metrics, or conditions rather than direct contradictions.

## Project Structure

```text
app.py
requirements.txt
.env.example
scripts/
  evaluate_scifact.py
  export_manual_labels.py
  evaluate_manual_labels.py
src/
  config.py
  schemas.py
  retrieval/semantic_scholar.py
  preprocessing/sentence_splitter.py
  claims/
  embeddings/
  clustering/
  nli/
  cards/
  evaluation/
  pipeline/run_pipeline.py
data/
  raw/semantic_scholar/
  processed/runs/
  evaluation/
```
