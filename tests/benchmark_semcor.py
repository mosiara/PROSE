import os
import sys
import spacy
import nltk
from nltk.corpus import semcor, wordnet as wn

# --- BULLETPROOF IMPORT BLOCK ---
# Dynamically points to your locked ~/src/prose/src/ structure
current = os.getcwd()
sys.path.insert(0, os.path.join(current, "src"))
# --------------------------------

nltk.download("semcor", quiet=True)
nltk.download("wordnet", quiet=True)
nlp = spacy.load("en_core_web_sm")

def evaluate_semcor(sample_target=50):
    from prose.pipeline import ProsePipeline
    pipeline = ProsePipeline()
    total_evaluated = 0
    preserved_gold = 0
    total_pure_csrr = 0.0
    stage_counts = {}
    unevaluable_skipped = 0

    print(f"Sampling real-world prepositional contexts from SemCor 3.0...")

    for sent in semcor.tagged_sents(tag="both"):
        words = []
        target_info = None

        for chunk in sent:
            is_target = hasattr(chunk, "label") and hasattr(chunk.label(), "synset")
            
            if hasattr(chunk, "leaves"):
                extracted_strings = [leaf[0] if isinstance(leaf, list) else leaf for leaf in chunk.leaves()]
            elif isinstance(chunk, list):
                extracted_strings = [str(chunk[0])]
            else:
                extracted_strings = [str(chunk)]
                
            if is_target:
                synset = chunk.label().synset()
                joined_target = "_".join(extracted_strings)
                
                if not target_info and synset.pos() == "n":
                    target_info = (joined_target, synset.name())
                    
            words.extend(extracted_strings)

        if not target_info:
            continue

        raw_sentence = " ".join(words)
        target_word, gold_synset = target_info

        doc = nlp(raw_sentence)
        is_governed_by_prep = False
        target_no_underscores = target_word.replace("_", " ")
        
        for token in doc:
            if token.text in target_no_underscores and token.head.pos_ == "ADP":
                is_governed_by_prep = True
                break
                
        if not is_governed_by_prep:
            continue 

        result = pipeline.process(doc.text, target_word, pos="NOUN")

        if gold_synset not in result.original_senses:
            unevaluable_skipped += 1
            continue

        # --- ISOLATE POS FROM SEMANTIC REDUCTION ---
        # Restrict the evaluation pools strictly to noun senses to eliminate POS-inflation
        pure_original_senses = [s for s in result.original_senses if ".n." in s]
        pure_surviving_senses = [s for s in result.surviving_senses if ".n." in s]
        
        pure_csrr = 1.0 - (len(pure_surviving_senses) / len(pure_original_senses)) if pure_original_senses else 0.0

        if len(pure_original_senses) > 1:
            total_evaluated += 1
            total_pure_csrr += pure_csrr
            stage_counts[result.stage_applied] = stage_counts.get(result.stage_applied, 0) + 1

            if gold_synset in pure_surviving_senses:
                preserved_gold += 1
            else:
                print(f"\n[⚠️ False Elimination] Target: '{target_word}' | Gold: {gold_synset}")
                print(f"Sentence: {raw_sentence}")
                print(f"Stage: {result.stage_applied}")
                print(f"Surviving ({len(pure_surviving_senses)}): {pure_surviving_senses}")

        if total_evaluated >= sample_target:
            break

    csrr_avg = (total_pure_csrr / total_evaluated) if total_evaluated > 0 else 0.0
    preservation_rate = (preserved_gold / total_evaluated) if total_evaluated > 0 else 0.0

    print("\n" + "=" * 50)
    print("📊 SEMCOR 3.0 EMPIRICAL BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Sentences Evaluated:     {total_evaluated}")
    print(f"Unevaluable Targets Skipped:   {unevaluable_skipped}")
    print(f"Pure Semantic Reduction Rate:  {csrr_avg:.1%} (CSRR)")
    print(f"Correct Sense Preservation:    {preservation_rate:.1%}")
    print("\nStage Execution Breakdown:")
    for stage, count in stage_counts.items():
        print(f"  - {stage}: {count} ({count/total_evaluated:.1%})")
    print("=" * 50)

if __name__ == "__main__":
    evaluate_semcor(sample_target=500)