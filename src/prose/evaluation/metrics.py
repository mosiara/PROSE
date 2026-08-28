"""
PROSE Evaluation Suite
Runs benchmarks against data/test_cases.json to calculate CSRR and gold sense preservation.
"""

import json
import os
from typing import Dict, Any, List
from src.prose.pipeline import prose_filter


def run_benchmark(test_cases_path: str = "data/test_cases.json") -> Dict[str, Any]:
    if not os.path.exists(test_cases_path):
        raise FileNotFoundError(f"Cannot find test file at {test_cases_path}")

    with open(test_cases_path, "r") as f:
        cases = json.load(f)

    results_summary = []
    total_original = 0
    total_surviving = 0
    gold_preserved_count = 0

    print(f"\n=======================================================")
    print(f"       RUNNING PROSE v0.1 BENCHMARK SUITE")
    print(f"=======================================================\n")

    for case in cases:
        c_id = case["id"]
        sentence = case["sentence"]
        target = case["target_token"]
        pos = case.get("target_pos", "NOUN")
        gold_synset = case.get("gold_synset")

        filter_res = prose_filter(sentence, target, pos)

        surviving_ids = [s.synset_id for s in filter_res.surviving_senses]
        gold_preserved = gold_synset in surviving_ids if gold_synset else True

        if gold_preserved:
            gold_preserved_count += 1

        total_original += len(filter_res.original_senses)
        total_surviving += len(filter_res.surviving_senses)

        results_summary.append({
            "id": c_id,
            "sentence": sentence,
            "target": target,
            "stage": filter_res.stage_applied,
            "csrr": filter_res.csrr,
            "gold_preserved": gold_preserved,
        })

        status_mark = "PASS" if gold_preserved else "FAIL"
        print(f"[{status_mark}] {c_id}")
        print(f"   Sentence: '{sentence}' (Target: '{target}')")
        print(f"   Applied:  {filter_res.stage_applied}")
        print(f"   CSRR:     {filter_res.csrr:.1%} reduction ({len(filter_res.original_senses)} -> {len(filter_res.surviving_senses)} senses)")
        print(f"   Gold ({gold_synset}) Preserved: {gold_preserved}\n")

    overall_csrr = (total_original - total_surviving) / total_original if total_original > 0 else 0.0
    preservation_rate = gold_preserved_count / len(cases) if cases else 0.0

    print("=======================================================")
    print(f"Overall Metrics across {len(cases)} test cases:")
    print(f" - Mean Candidate Sense Reduction Rate (CSRR): {overall_csrr:.1%}")
    print(f" - Correct Sense Preservation Rate:            {preservation_rate:.1%}")
    print("=======================================================\n")

    return {
        "overall_csrr": overall_csrr,
        "preservation_rate": preservation_rate,
        "case_details": results_summary,
    }


if __name__ == "__main__":
    run_benchmark()
