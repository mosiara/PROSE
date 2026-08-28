# 📖 PROSE: Pre-processing Reduction of Senses Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP: spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![Lexicon: WordNet](https://img.shields.io/badge/Lexicon-WordNet-green.svg)](https://nltk.org/)

**PROSE is a deterministic structural pre-processing layer designed to reduce the candidate sense space before it is presented to a Large Language Model.**

It is an upstream candidate-sense filter, not a standalone Word Sense Disambiguation (WSD) solver. By mapping explicit grammatical relationships to lexical constraints, PROSE deterministically eliminates candidate senses that are incompatible with the resulting structural and lexical constraints before the downstream LLM evaluates them. When the available evidence is insufficient to support elimination, PROSE fails open and preserves the full candidate set.

---

## 🧠 Core Concept: The "Executive Secretary" Model

When an LLM performs WSD, a pipeline may provide the model with multiple candidate dictionary senses for a target word. Depending on the sense inventory, this can introduce irrelevant candidates and increase the amount of context the downstream model must evaluate.

PROSE acts as an executive secretary: it examines explicit structural relationships in the sentence, applies deterministic lexical constraints, and removes candidate senses that are incompatible with those constraints. The downstream LLM remains the final semantic decision-maker.

Rather than rewriting, summarizing, or embedding the source sentence, PROSE operates on the candidate sense set:

Raw sentence + candidate senses -> structural analysis -> conservative candidate pruning -> downstream LLM evaluation.

If the structural and lexical constraints do not justify elimination, PROSE fails open and passes the complete candidate set downstream.

~~~text
[ Raw Sentence + WordNet Candidate Senses ]
              │
              ▼
   ┌──────────────────────┐
   │    PROSE Pipeline    │
   │  (spaCy + WordNet)   │
   └──────────┬───────────┘
              │ Prunes candidates incompatible with structural and lexical constraints
              ▼
[ Filtered Candidates (Reduced Search Space) ]
              │
              ▼
   ┌────────────────────┐
   │ Downstream LLM WSD │ ──► Final Disambiguated Sense
   └────────────────────┘
~~~

---

## ⚙️ Multi-Stage Architecture

PROSE operates under a strict **"fail open"** constraint. It prioritizes candidate preservation over aggressive filtering. If the available structural and lexical evidence do not support a deterministic elimination, PROSE fails open and preserves the full candidate set for the downstream model.

| Stage | Mechanism | Deterministic Rule / Mapping |
| :--- | :--- | :--- |
| **Stage 1: Direct Constraints** | Preposition-to-Domain Mapping | *Preposition -> Compatible Lexical Domains.* Uses supported prepositional constructions to constrain candidate senses according to compatible WordNet lexicographer domains. The engine includes explicit mappings for 30+ supported English prepositions to subsets of compatible WordNet lexicographer domains. |
| **Stage 2: Affordance Traversal** | Syntactic Attachment + Lexical Affordances | *Dependency Arc -> Affordance Validation.* Resolves ambiguous attachments by combining deep dependency structures (identifying the governing verb and its active/passive subject) with lexical affordance checks (e.g., Animate Subject + "with" = Comitative). |
| **Stage 3: Safety Fallback** | Conservative Preservation Guard | If the available structural and lexical evidence does not support a deterministic elimination, PROSE fails open and preserves the full candidate set for the downstream model. |

---

## 📊 Empirical Benchmarks (v0.3 Release)

Validated on an evaluated sample of 50 SemCor 3.0 sentences and downstream generative LLMs (`gemini-3.6-flash`). 

* **Correct Sense Preservation Rate (CSP):** `100.0%` (0 false eliminations across the evaluated sample. Designed to prioritize safety and fail open).
* **Mean Candidate Sense Reduction Rate (CSRR):** `6.8%` (Noun senses only, eliminating POS-filtering inflation to establish a rigorous baseline reduction rate).
* **LLM Prompt Token Reduction:** `57.7%` (Downstream ablation testing confirms the structural candidate filtering translates directly into substantial prompt token savings for generative models, reducing token overhead by over 50%).
* **Downstream Accuracy Degradation:** `0.0%` (The filtered candidate space maintained baseline LLM WSD accuracy with zero observed degradation in the test sample).

---

## 🚀 Quickstart

### 1. Installation (PEP 621)

PROSE is configured as a standard, installable Python module.

~~~bash
git clone https://github.com/mosiara/PROSE.git
cd PROSE
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m spacy download en_core_web_sm
~~~

### 2. The Production Simulator (Custom Testing)
Test your own idioms and sentences to watch the filter reduce the candidate pool in real-time.

~~~bash
python3 -m tests.test_custom
~~~

### 3. Run the SemCor 3.0 Benchmark
Reproduce the baseline syntactic metrics (CSRR and CSP) on the SemCor evaluation sample.

~~~bash
python3 -m tests.benchmark_semcor
~~~

### 4. Downstream LLM Ablation Test
Verify prompt token reduction and accuracy parity using a live generative model. *(Requires a `GEMINI_API_KEY` in your `.env` file)*.

~~~bash
python3 -m tests.ablation_test
~~~

---

## 📂 Project Architecture

~~~text
PROSE/
├── .gitignore
├── LICENSE
├── README.md
├── pipeline.py                      # Root-level API interface
├── pyproject.toml                   # PEP 621 Package configuration
├── requirements.txt
├── src/
│   └── prose/
│       ├── core/
│       │   ├── filter.py            # Multi-stage elimination logic & Animacy validation
│       │   ├── lexical.py           # WordNet candidate extraction
│       │   └── parser.py            # spaCy dependency parsing & subject extraction
│       └── evaluation/
│           └── pipeline.py          # Evaluation pipeline module
└── tests/
    ├── ablation_test.py             # LLM prompt token and accuracy comparison
    ├── benchmark_semcor.py          # Empirical grading script (Preservation & CSRR)
    └── test_custom.py               # Real-time candidate reduction simulator
~~~

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.