# PROSE: Pre-processing Reduction of Senses Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP: spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![Lexicon: WordNet](https://img.shields.io/badge/Lexicon-WordNet-green.svg)](https://nltk.org/)

**PROSE** is a deterministic, neuro-symbolic middleware layer that enforces the mathematical structure of syntax to filter lexical ambiguity before neural evaluation.

Modern Large Language Models (LLMs) treat Word Sense Disambiguation (WSD) as a probabilistic guessing game, ingesting massive lists of dictionary definitions and relying on attention mechanisms to weigh them. PROSE rejects this brute-force approach. By exploiting strict syntactic dependency trees and the thematic roles assigned by relational words (like prepositions and verbs), PROSE mathematically eliminates impossible dictionary senses upstream.

We do not parse for context; we parse for structural physics. If a noun violates the syntactic constraints of its environment, it is dropped.

## 🔬 The Core Hypothesis

Language is a mathematical structure. The relational grammar of a sentence dictates the valid semantic domains of its constituents.

Our underlying hypothesis is that **syntactic dependency arcs encode deterministic semantic boundary conditions (affordances, selectional preferences, and thematic roles) that can safely prune candidate noun senses prior to probabilistic neural inference.**

If a target noun is the object of the preposition "with" governed by an animate subject, the syntactic math dictates it is fulfilling a Comitative (companion) or Instrument role. A noun sense categorizing the target as a `noun.time` or `noun.location` is structurally impossible and can be deterministically eliminated.

---

## Architecture

PROSE operates as an "Executive Secretary" pre-filter, acting as the bridge between a syntactic parser (spaCy) and a lexical ontology (WordNet). It traverses the dependency tree rather than reading left to right, mapping actual
parent–child grammatical relationships.

```
Sentence + WordNet candidate senses
                |
                v
        PROSE Pipeline
        (spaCy + WordNet)
                |
                |  eliminates senses incompatible with the structural constraint
                v
        Reduced candidate set
                |
                v
        Downstream LLM WSD  --->  final sense
```


1. **Syntactic Extraction (The Math of the Sentence):** PROSE traverses the strict dependency tree, triangulating the Target Noun, the Relational Word (preposition, conjunction, or direct verb), and the Upstream Head. It does not rely on proximity or linear token matching; it maps the definitive grammatical arc.
2. **Thematic Role Assignment:** Based on established linguistic frameworks, the engine maps the extracted syntactic relation to a specific thematic role (e.g., Locus, Agent, Instrument, State).
3. **Domain Constraint Filtering:** PROSE cross-references the target's WordNet candidate senses (`lexnames`) against the allowed semantic domains for that thematic role.
4. **Fail-Open Safety:** PROSE is a conservative shield. If a target noun lacks a definitive relational anchor, or if the filtered intersection is empty, the engine triggers a fail-open sequence and preserves the entire candidate list to protect downstream accuracy.

## ⚙️ The Multi-Stage Filter

**The Multi-Stage Filter (Module A)**

Currently, PROSE evaluates prepositions as the primary relational anchors utilizing a multi-stage triage protocol:

* **Stage 1 (Direct Constraints):** Evaluates strict spatial, temporal, and abstract domains driven by broad prepositions (e.g., *in*, *on*, *at*, *from*) by mapping them to compatible WordNet lexicographer domains.
* **Stage 2 (Affordance Traversal):** Resolves ambiguous prepositions by evaluating dynamic thematic roles requiring upstream syntactic verification:
* **Comitative/Instrument:** Triggers on *with* governed by a verbal governor, differentiating based on animate vs. inanimate subjects.

* **Passive Agent:** Triggers on *by* when governed by passive-voice verb morphology.

* **Means/Proximity:** Triggers on *by* in active-voice verbal constructs.

* **Stage 3 (Fail Open):** Evidence insufficient → full candidate set preserved to act as an absolute safety net.



---

# 📊 Evaluation & Metrics

PROSE is evaluated strictly on its structural precision, not on commercial downstream byproducts. The engine's success is measured by two competing scientific metrics across the SemCor 3.0 empirical benchmark:

1. **Candidate Sense Reduction Rate (CSRR):** The percentage of pure semantic noise deterministically eliminated from the hypothesis space (Macro/Micro).
2. **Gold Sense Preservation Rate (GPR):** The percentage of trials where the true, human-annotated definition safely survives the structural filter.

*Target Baseline:* PROSE is engineered to achieve a mathematically defensible pure CSRR while maintaining a GPR of ≥ 98.0%, proving that syntactic constraints can successfully isolate meaning without blinding downstream systems.

## 🚀 Roadmap

* **Filter A (Complete):** Prepositional thematic roles (`pobj`).
* **Filter B (In Development):** Upstream/Downstream Verb Relations (Direct Objects `dobj`, Subjects `nsubj`).
* **Filter C (Planned):** Multi-Word Expression (MWE) and Idiom Bypass shielding.

---

*PROSE: Because language has rules, and neural networks shouldn't have to guess them.*
