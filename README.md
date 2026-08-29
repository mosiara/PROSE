# PROSE: Pre-processing Reduction of Senses Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NLP: spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![Lexicon: WordNet](https://img.shields.io/badge/Lexicon-WordNet-green.svg)](https://nltk.org/)

**Prepositions are the defenders of context. PROSE uses them to eliminate impossible
noun senses before they ever reach a Large Language Model.**

A noun on its own is inert. *Bank* could be anything. What situates it is its relation
to everything around it — and prepositions are the words that carry that relation
explicitly. PROSE reads those relations and deletes the senses they rule out.

It is not a Word Sense Disambiguation system. It never selects a sense. It shortens the
list the LLM has to choose from, and when the evidence doesn't justify a cut, it passes
everything through untouched.

---

## The Core Hypothesis

**Prepositions are the defenders of context.**

A noun on its own is inert. *Bank* could be anything. What situates it is its relation
to everything around it — and prepositions are the words that carry that relation
explicitly. They act as operators that assign thematic roles (Instrument, Locative,
Comitative, Agent) to the nouns they govern, and those roles constrain what the noun
can mean far more tightly than the noun constrains itself.

Current pipelines hand an LLM up to twenty raw dictionary definitions for one target
word and rely on attention to weigh them. PROSE holds that some of those definitions
are structurally unavailable before any weighing begins: if a sense contradicts the
thematic role its relational anchor assigns, it can be eliminated deterministically —
same input, same output, no learning, no statistics.

**The invariant.** Delete only when the evidence supports it. When it doesn't, keep
everything. Preservation of the correct sense is not a metric to maximize; it is a
constraint the system must not violate. Reduction is whatever remains achievable
under that constraint.

---

## Architecture

PROSE traverses the dependency tree rather than reading left to right, mapping actual
parent–child grammatical relationships.

```
[ Sentence + WordNet candidate senses ]
              │
              ▼
   ┌──────────────────────┐
   │    PROSE Pipeline    │
   │  (spaCy + WordNet)   │
   └──────────┬───────────┘
              │  eliminates senses incompatible with the structural constraint
              ▼
[ Reduced candidate set ]
              │
              ▼
   ┌────────────────────┐
   │ Downstream LLM WSD │ ──► final sense
   └────────────────────┘
```

The LLM is never told how the list was shortened. PROSE shows the table, not the saw.

### Filter Modules

| Module | Scope | Status |
|---|---|---|
| **A — Prepositional Relations** | Nouns as prepositional objects (`pobj`). Maps prepositional operators to thematic roles and constrains the governed noun. | **Implemented** |
| **B — Direct Relational Anchors** | Direct objects and subjects constrained by the main clause verb (`dobj`, `nsubj`). | Planned |
| **C — Idiom Fail-Safe** | Bypass layer for fixed phrases (*in the dark*, *by the way*) so non-compositional language never reaches the constraint engine. | Planned |

### The three stages (Module A)

| Stage | Mechanism |
|---|---|
| **1 — Direct Constraint** | Preposition → compatible WordNet lexicographer domains. |
| **2 — Affordance Traversal** | Ambiguous prepositions resolved by dependency structure: *with* + verbal governor + animate subject → Comitative/Instrument; *by* + passive governor → Agent; *by* + active verbal governor → Means/Proximity. |
| **3 — Fail Open** | Evidence insufficient → full candidate set preserved. |

---

## Empirical Status

Every number below comes from `h1_signal_test.py` in this repository, run on the full
SemCor 3.0 corpus, commit `e904e5b`. The script does not invoke the filter — it measures
the corpus directly.

### Coverage — what PROSE can reach

Pending. Coverage figures from the full-corpus run have not yet been recorded here.

### The oracle ceiling — the best any lexname filter could do

Keeping only senses that share the gold sense's lexicographer domain — cheating by
consulting the answer key — yields:

| Population | Mean CSRR | Median |
|---|---|---|
| All polysemous targets | 52.7% | 56.2% |
| `pobj`-governed targets | **54.1%** | **60.0%** |

**Roughly half the candidate list is removable in principle.** The approach is not
ceiling-bound.

### Signal — do prepositions actually constrain?

Comparing each preposition's gold-lexname distribution against the distribution over
all nouns (Resnik-style selectional preference strength). Prior top-3 mass: **0.312**.

| Preposition | n | Top-3 lexname mass | KL vs prior |
|---|---|---|---|
| per | 96 | 0.82 | 2.143 |
| during | 134 | 0.84 | 1.656 |
| down | 47 | 0.83 | 1.279 |
| across | 78 | 0.77 | 1.217 |
| around | 52 | 0.75 | 1.103 |
| below | 31 | 0.68 | 1.090 |
| … | | | |
| in | 4,867 | 0.33 | 0.172 |
| with | 1,697 | 0.33 | 0.112 |
| by | 1,079 | 0.36 | 0.084 |
| to | 2,148 | 0.34 | 0.072 |
| of | 8,327 | 0.32 | **0.024** |

**The signal is real and unevenly distributed.** Prepositions that retain spatial or
temporal meaning (*during*, *across*, *below*, *per*) concentrate their objects far
above the base rate. Prepositions bleached into grammatical glue (*of*, *to*, *by*)
are indistinguishable from it — *of* governs its objects almost exactly as English
distributes nouns generally.

Grouped by strength:

| Band | Prepositions | Share of admissions |
|---|---|---|
| KL ≥ 1.0 | 6 | 1.6% |
| KL 0.4–1.0 | 15 | 8.4% |
| KL < 0.4 | 17 | **90.1%** |

Ninety percent of the volume comes from prepositions carrying little or no constraint.

---

## Known Limitations

Stated plainly, because a filter that overstates its own guarantees is worse than one
that filters less.

**The fail-open guard is incomplete.** It restores the full candidate set only when the
constraint eliminates *every* candidate. A non-empty but incorrect surviving set is
accepted, which can silently drop the gold sense. This does not yet satisfy the
invariant above, and closing it is the current priority.

**No certainty condition on elimination.** The filter treats every preposition
identically. Given the divergence table, it currently deletes as readily on *of*
(KL 0.024) as on *during* (KL 1.656).

**Preposition coverage.** Stage 1 currently maps 16 prepositions, down from 39.
Several high-signal prepositions (*across*, *below*, *near*, *along*, *off*) are among
those absent.

**Historical benchmark figures are not reported here.** Earlier releases quoted
preservation and reduction rates obtained after repeated tuning against the same
evaluation corpus. Those are development-set numbers and have been withdrawn pending a
held-out evaluation.

**Subject animacy is a coarse proxy.** Currently `PRON`/`PROPN` only, so common-noun
subjects are treated as inanimate.

---

## Evaluation Metrics

PROSE is evaluated on structural precision, not on downstream cost savings.

- **CSRR** (Candidate Sense Reduction Rate) — proportion of the candidate list
  eliminated, macro and micro.
- **GPR** (Gold Sense Preservation Rate) — proportion of trials where the human-
  annotated sense survives. Reported conditional and unconditional, with the
  denominator stated.
- **Coverage** — proportion of targets where any rule fires.
- **Oracle ceiling** — maximum achievable reduction at perfect precision.

Prompt-token and cost metrics are deliberately excluded. They are a byproduct, not a
result, and quoting them invited claims the code did not support.

---

## Quickstart

```bash
cd PROSE
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python3 -m spacy download en_core_web_sm
```

**Signal test and oracle ceiling** — measures the corpus directly; does not invoke
the filter:
```bash
python3 src/prose/evaluation/h1_signal_test.py
python3 src/prose/evaluation/h1_signal_test.py --limit 500   # quick run
```

Test scripts under `tests/` are run directly; see the file headers for usage.

---

## Roadmap

1. Close the fail-open invariant — reject non-empty surviving sets that cannot be
   justified, not merely empty ones.
2. Gate elimination on measured constraint strength; fail open below threshold.
3. Restore high-signal prepositions to Stage 1.
4. Matched random-deletion control at equal reduction rate.
5. Held-out evaluation split, run once, against a frozen taxonomy.
6. Filter Module B — verb-governed subjects and direct objects.
7. Filter Module C — idiom bypass.

---

## License

MIT. See `LICENSE`.

---

*PROSE: language has structure, and neural networks shouldn't have to guess it.*
