"""
PROSE Downstream LLM Ablation Test (Hardened Native Implementation)
"""

import time
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


from prose.core.filter import run_multistage_filter
from prose.core.lexical import extract_candidate_senses

# Load environment variables from .env
load_dotenv()

SYNSET_REGEX = re.compile(r'\b[a-zA-Z0-9_\-]+\.[nvar]\.\d{2}\b')

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def query_llm_with_retry(prompt: str, client: genai.Client) -> dict:
    """
    Executes the prompt against Gemini using the native SDK with exponential backoff retries.
    """
    time.sleep(1)  # Base polite pacing
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        )
    )

    raw_text = ""
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if part.text:
                raw_text += part.text
    if not raw_text and hasattr(response, 'text') and response.text:
        raw_text = response.text

    # Extract rigid synset identifier via regex
    match = SYNSET_REGEX.search(raw_text)
    extracted_synset = match.group(0) if match else "UNPARSED"

    prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else len(prompt.split())

    return {
        "raw_text": raw_text.strip(),
        "extracted_synset": extracted_synset,
        "prompt_tokens": prompt_tokens
    }

def build_wsd_prompt(sentence: str, target: str, candidate_senses: list) -> str:
    """
    Constructs a rigid, repeatable prompt for the WSD task with zero trailing padding.
    """
    senses_text = "\n".join([f"- {sense.synset_id}: {sense.definition}" for sense in candidate_senses])

    return (
        f"You are an expert linguist. Disambiguate the target word in the given sentence.\n\n"
        f"Sentence: \"{sentence}\"\n"
        f"Target Word: \"{target}\"\n\n"
        f"Available Senses:\n"
        f"{senses_text}\n\n"
        f"Identify the single best sense from the list above that matches the target word in the context of the sentence.\n"
        f"Output ONLY the exact sense name (e.g., 'word.n.01'). Do not include any other text or explanation."
    )

def run_ablation_test():
    print("🚀 Starting LLM Ablation Test")
    print("Comparing Unfiltered WordNet vs. PROSE-Filtered Candidate Lists\n")

    client = genai.Client()
    
    import spacy
    nlp = spacy.load("en_core_web_sm")

    test_cases = [
        {
            "sentence": "Toying with her field in the early stages , Garden Fresh was asked for top speed only in the stretch by Jockey Philip Grimm.",
            "target": "field",
            "gold": "field.n.12"
        },
        {
            "sentence": "He merely became victimized by a form of athletics that respects no one.",
            "target": "form",
            "gold": "kind.n.01"
        }
    ]

    results = {
        "total_evaluated": 0,
        "unfiltered_correct": 0,
        "filtered_correct": 0,
        "unfiltered_tokens": 0,
        "filtered_tokens": 0
    }

    for case in test_cases:
        sentence_str = case["sentence"]
        target_word = case["target"]
        gold_synset = case["gold"]

        all_candidates = extract_candidate_senses(target_word)
        if len(all_candidates) < 2:
            continue

        doc = nlp(sentence_str)
        filter_result = run_multistage_filter(doc, target_word)
        surviving_candidates = [c for c in all_candidates if c.synset_id in filter_result.surviving_senses]

        unfiltered_prompt = build_wsd_prompt(sentence_str, target_word, all_candidates)
        filtered_prompt = build_wsd_prompt(sentence_str, target_word, surviving_candidates)

        try:
            unfiltered_resp = query_llm_with_retry(unfiltered_prompt, client)
            filtered_resp = query_llm_with_retry(filtered_prompt, client)
        except Exception as err:
            print(f"Failed evaluation for target '{target_word}' after retries: {err}")
            continue

        results["unfiltered_tokens"] += unfiltered_resp["prompt_tokens"]
        results["filtered_tokens"] += filtered_resp["prompt_tokens"]

        if unfiltered_resp["extracted_synset"] == gold_synset:
            results["unfiltered_correct"] += 1
        if filtered_resp["extracted_synset"] == gold_synset:
            results["filtered_correct"] += 1

        results["total_evaluated"] += 1

        print(f"Target: '{target_word}' | Gold: {gold_synset}")
        print(f"  - Unfiltered -> LLM Picked: {unfiltered_resp['extracted_synset']} | Tokens: {unfiltered_resp['prompt_tokens']}")
        print(f"  - Filtered   -> LLM Picked: {filtered_resp['extracted_synset']} | Tokens: {filtered_resp['prompt_tokens']}\n")

    print("-" * 50)
    print("📊 ABLATION TEST RESULTS")
    print("-" * 50)

    if results["total_evaluated"] > 0:
        unf_acc = (results["unfiltered_correct"] / results["total_evaluated"]) * 100
        fil_acc = (results["filtered_correct"] / results["total_evaluated"]) * 100

        print("Baseline A (Unfiltered):")
        print(f"  - Accuracy: {unf_acc:.1f}% ({results['unfiltered_correct']}/{results['total_evaluated']})")
        print(f"  - Total Input Tokens: {results['unfiltered_tokens']}")
        print("\nBaseline B (PROSE-Filtered):")
        print(f"  - Accuracy: {fil_acc:.1f}% ({results['filtered_correct']}/{results['total_evaluated']})")
        print(f"  - Total Input Tokens: {results['filtered_tokens']}")
        
        token_savings = 0.0
        if results["unfiltered_tokens"] > 0:
            token_savings = ((results["unfiltered_tokens"] - results["filtered_tokens"]) / results["unfiltered_tokens"]) * 100
        print(f"\nNet Token Reduction: {token_savings:.1f}%")
    print("-" * 50)

if __name__ == "__main__":
    run_ablation_test()