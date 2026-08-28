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

Raw sentence + candidate senses $\rightarrow$ structural analysis $\rightarrow$ conservative candidate pruning $\rightarrow$ downstream LLM evaluation.

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
| **Stage 1: Direct Constraints** | Preposition-to-Domain Mapping | *Preposition &rarr; Compatible Lexical Domains.* Uses supported prepositional constructions to constrain candidate senses according to compatible WordNet lexicographer domains. The current prototype includes explicit mappings for 30+ supported English prepositions to subsets of compatible WordNet lexicographer domains. |
| **Stage 2: Affordance Traversal** | Syntactic Attachment + Lexical Affordances | *Dependency Arc &rarr; Affordance Validation.* Resolves ambiguous attachments by combining dependency structure with lexical affordance checks, such as whether a noun is compatible with an instrument or attribute interpretation. |
| **Stage 3: Safety Fallback** | Conservative Preservation Guard | If the available structural and lexical evidence does not support a deterministic elimination, PROSE fails open and preserves the full candidate set for the downstream model. |

---

## 📊 Empirical Benchmark (v0.1 Prototype)

Validated on an evaluated sample of 50 SemCor 3.0 sentences. Cases for which the SemCor annotation could not be unambiguously aligned with the prototype's candidate sense inventory were excluded before evaluation. SemCor is a widely used manually sense-annotated corpus for Word Sense Disambiguation evaluation.

* **Correct Sense Preservation Rate:** `100.0%` (0 false eliminations across the evaluated 50-sentence sample. Designed to prioritize safety, though real-world performance depends on syntactic parsing accuracy).
* **Mean Candidate Sense Reduction Rate (CSRR):** `17.4%` (Average candidate reduction observed within the evaluated test set).
* **Filter Activation:** `50.0%` of valid targets triggered at least one deterministic filtering rule. A valid target is an evaluable target word with an unambiguous candidate-sense alignment and a structural configuration supported by the prototype.

---

## 🚀 Quickstart

### 1. Installation

~~~bash
git clone https://github.com/mosiara/PROSE.git
cd PROSE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
~~~

### 2. The Production Simulator (Custom Testing)
Test your own idioms and sentences to watch the filter reduce the candidate pool in real-time.

~~~bash
python3 -m tests.test_custom
~~~

### 3. Run the SemCor 3.0 Benchmark
Reproduce the baseline metrics and preservation rate on the SemCor evaluation sample.

~~~bash
python3 -m tests.benchmark_semcor
~~~

---

## 📂 Project Architecture

~~~text
PROSE/
├── src/
│   └── prose/
│       ├── core/
│       │   ├── lexical.py       # WordNet candidate extraction
│       │   ├── parser.py        # spaCy dependency parsing layer
│       │   └── filter.py        # Multi-stage elimination logic & Preposition Maps
│       └── pipeline.py          # Top-level API interface
├── tests/
│   ├── benchmark_semcor.py      # Empirical grading script (Preservation & CSRR)
│   └── test_custom.py           # Real-time candidate reduction simulator
├── requirements.txt
└── README.md
~~~

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
