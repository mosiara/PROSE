import os
import sys
import spacy

# --- BULLETPROOF IMPORT BLOCK ---
current = os.getcwd()
sys.path.extend([current, os.path.join(current, 'src'), os.path.join(current, 'src', 'prose')])
from core.filter import run_multistage_filter
# --------------------------------

nlp = spacy.load("en_core_web_sm")

# --- YOUR VICTORY LAP ---
sentence = "Alice caught the butterfly in the net."
target = "net"
# ------------------------

print(f"\n🧠 PROSE Cognitive Offloader Activated")
print(f"Analyzing: '{sentence}'")
print(f"Targeting: '{target}'...\n")

doc = nlp(sentence)
result = run_multistage_filter(doc, target_word=target)

print(f"🛡️ Structural Shield Deployed: {result.stage_applied}")
print(f"Original Senses: {len(result.original_senses)}")
print(f"Surviving Senses: {len(result.surviving_senses)}")
print(f"Compute Drag Reduced By: {result.csrr:.1%}\n")
print("Surviving Definitions Passed to LLM:")
for sense in result.surviving_senses:
    print(f" - {sense}")
print("\n")