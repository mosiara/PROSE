import random
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
from nltk.tree import Tree
from prose.pipeline import prose_filter
from prose.core.lexical import extract_candidate_senses

def estimate_tokens(text: str) -> int:
    """
    Approximates tokens (1 token ≈ 4 characters in English) to calculate 
    absolute savings without requiring external dependencies like tiktoken.
    """
    return len(text) // 4

def build_llm_prompt(sentence: str, target: str, senses: list) -> str:
    """
    Constructs a standard zero-shot WSD prompt simulating exactly 
    what the downstream LLM will evaluate.
    """
    prompt = f"Context: {sentence}\nTarget word: '{target}'\n\nOptions:\n"
    for i, synset_id in enumerate(senses):
        try:
            syn = wn.synset(synset_id)
            prompt += f"{i+1}. {synset_id}: {syn.definition()}\n"
        except:
            prompt += f"{i+1}. {synset_id}: (Definition unavailable)\n"
    prompt += "\nWhich option best fits the context? Output only the ID."
    return prompt

def run_ablation(sample_size: int = 100):
    print(f"Sampling {sample_size} NOUN targets from SemCor for ablation testing...\n")
    
    valid_cases = []
    # Collect a deep pool of noun targets to sample from
    for sent in semcor.tagged_sents(tag="sem"):
        words = []
        target_info = None
        
        for node in sent:
            if isinstance(node, Tree):
                # Using space to join multi-word expressions naturally for the LLM prompt
                word_str = " ".join(node.leaves())
                words.append(word_str)
                try:
                    # Extract WordNet synset from the SemCor tree label
                    lemma = node.label()
                    if hasattr(lemma, 'synset'):
                        synset = lemma.synset()
                        if synset and synset.pos() == 'n':
                            target_info = (word_str, synset.name())
                except Exception:
                    pass
            elif isinstance(node, list):
                # NLTK untagged words are returned as lists of strings
                words.extend(node)
            else:
                words.append(str(node))
                
        if target_info:
            sentence_str = " ".join(words)
            valid_cases.append((sentence_str, target_info[0], target_info[1]))
            if len(valid_cases) >= sample_size * 5:
                break
                
    samples = random.sample(valid_cases, sample_size)
    
    total_baseline_tokens = 0
    total_filtered_tokens = 0
    gold_retained = 0
    
    for sentence, target, gold_sense in samples:
        # 1. Unfiltered Baseline (POS strictly enforced to NOUN)
        baseline_senses = [s.synset_id for s in extract_candidate_senses(target, pos="NOUN")]
        baseline_prompt = build_llm_prompt(sentence, target, baseline_senses)
        total_baseline_tokens += estimate_tokens(baseline_prompt)
        
        # 2. PROSE Filtered (POS strictly enforced to NOUN)
        result = prose_filter(sentence, target, pos="NOUN")
        filtered_prompt = build_llm_prompt(sentence, target, result.surviving_senses)
        total_filtered_tokens += estimate_tokens(filtered_prompt)
        
        # 3. Accuracy Retention Evaluation
        if gold_sense in result.surviving_senses:
            gold_retained += 1

    # Calculate Final Metrics
    absolute_savings = total_baseline_tokens - total_filtered_tokens
    percent_savings = (absolute_savings / total_baseline_tokens) * 100 if total_baseline_tokens else 0
    accuracy_retention = (gold_retained / sample_size) * 100
    
    print("==================================================")
    print("🚀 PROSE LLM ABLATION TEST RESULTS")
    print("==================================================")
    print(f"Sample Size:           {sample_size} sentences")
    print(f"Baseline Prompt Size:  {total_baseline_tokens:,} tokens")
    print(f"Filtered Prompt Size:  {total_filtered_tokens:,} tokens")
    print(f"Absolute Savings:      {absolute_savings:,} tokens")
    print(f"Token Reduction:       {percent_savings:.1f}%")
    print(f"Accuracy Retention:    {accuracy_retention:.1f}% (Gold sense preserved)")
    print("==================================================")

if __name__ == "__main__":
    run_ablation(sample_size=100)