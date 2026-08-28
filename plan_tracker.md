# PROSE Project Plan & Tracking

## Phase 1: Engine Expansion & Benchmarking (Current)

### Step 1: Core Engine Update
* **Action:** Update `src/prose/core/filter.py` with the expanded preposition taxonomy (Locative, Temporal, Agentive, etc.).
* **Justification:** The engine needs the actual rules embedded before it can test them against real-world data.
* **Rel:** Upgrades PROSE from a narrow prototype to a comprehensive prepositional filter.
* **Status:** In Progress 

### Step 2: Benchmark Harness Creation
* **Action:** Create `tests/benchmark_semcor.py` to evaluate the engine against the SemCor 3.0 corpus.
* **Justification:** We need an empirical script to calculate Candidate Sense Reduction Rate (CSRR) and preservation accuracy on messy, real-world sentences.
* **Rel:** Provides the concrete proof and metrics needed to validate the PROSE hypothesis to the NLP community.
* **Status:** Pending

### Step 3: Empirical Execution
* **Action:** Run the benchmark script and analyze the terminal output for false eliminations.
* **Justification:** Real-world data will expose edge cases (like idioms or passive voice) that our rules might mishandle.
* **Rel:** Identifies exactly which heuristics need tuning before we package PROSE as a native spaCy component.
* **Status:** Pending