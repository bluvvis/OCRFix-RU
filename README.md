# OCRFix-RU

**2.3 Hybrid Word+Character N-gram for Error Correction**

Hybrid word+character n-gram project for OCR error correction.

The repository is aligned with the case study:
**Hybrid Word+Character N-gram for Error Correction**  
Combine character-level and word-level n-grams to detect and auto-correct noisy OCR or social-media text, evaluate on synthetic noise, and compare against a vanilla word n-gram baseline.

## What is implemented

- Character-level and word-level n-gram language models.
- Hybrid correction pipeline with candidate generation and reranking.
- Synthetic OCR noise generator.
- Evaluation module that compares:
  - `word-only` baseline
  - `hybrid word+char` approach
- Additional metrics: `WER` and `CER`.
- Ablation runner for `alpha` and noise-rate sweeps.
- Pytest suite for core logic and regression checks.
- CLI demo for quick local runs.
- Dataset loader + optional corpus downloader.
- Auto-generation of case-study artifacts:
  - `artifacts/report.json`
  - one-page LaTeX poster source in `poster/poster.tex`
  - poster PDF: [`poster/NLP_poster.pdf`](poster/NLP_poster.pdf)
  - structured experiment notebook in `notebooks/case_study.ipynb`

## Project layout

- `src/ocrfix_ru/noise.py` - synthetic OCR corruption.
- `src/ocrfix_ru/lm.py` - word and char n-gram language models.
- `src/ocrfix_ru/corrector.py` - hybrid corrector.
- `src/ocrfix_ru/evaluate.py` - benchmark and metrics.
- `src/ocrfix_ru/dataset.py` - corpus download/loading utilities.
- `src/ocrfix_ru/poster.py` - one-page LaTeX poster generator.
- `src/ocrfix_ru/cli.py` - command line interface.
- `scripts/build_case_study.py` - end-to-end artifact builder.
- `poster/poster.tex` - generated poster source.
- `poster/NLP_poster.pdf` - final one-page poster PDF.
- `notebooks/case_study.ipynb` - case-study notebook (motivation, protocol, metrics, headline run, qualitative demo, ablations).
- `tests/` - automated tests.

## Setup

```bash
python -m pip install -e .[dev]
```

## Run tests

```bash
python -m pytest
```

## Run benchmark

```bash
python -m ocrfix_ru.cli --evaluate
```

## Build full case-study artifacts

```bash
python scripts/build_case_study.py
```

This command downloads a Russian corpus (if absent), runs evaluation + ablations, and generates:

- `artifacts/report.json`
- `poster/poster.tex` (1-page poster source)
- data file in `data/raw/russian_corpus.txt`
- `poster/poster.pdf` (if `pdflatex` is installed)

Main headline metric in `artifacts/report.json` is computed on a controlled synthetic OCR setup (`seed=42`, `error_rate=0.30`, `alpha=0.90`) to directly report gain over vanilla word n-gram baseline.

## Dataset and training/evaluation protocol

- External corpus source for ablations: Project Gutenberg (`https://www.gutenberg.org/cache/epub/2554/pg2554.txt`), saved to `data/raw/russian_corpus.txt`.
- Corpus preprocessing:
  - sentence split by punctuation,
  - trim/normalize whitespace,
  - lowercase,
  - filter short sentences (`min_words=3`),
  - cap to `max_sentences=1800`.
- Model training in this project means fitting count-based n-gram LMs:
  - `WordNGramLM` and `CharNGramLM` are fitted by counting n-grams from training texts,
  - no neural optimizer/backprop is used.
- Main report (`main_report`) uses controlled synthetic OCR setup:
  - `seed=42`, `sample_size=80`, `error_rate=0.30`, `alpha=0.90`.
- Ablation report (`ablation`) uses corpus-derived sentence set with grid:
  - seeds `[7, 11, 19]`,
  - error rates `[0.1, 0.2, 0.3]`,
  - alphas `[0.0, 0.3, 0.7, 1.0]`,
  - `sample_size=24`.

## Run correction demo

```bash
python -m ocrfix_ru.cli --text "мама мыла раму и к0т спит"
```
