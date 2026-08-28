"""
PROSE Dependency Parsing Layer
Extracts prepositional attachments and governing syntactic heads using spaCy.
"""

import spacy
from typing import Optional
from pydantic import BaseModel

nlp = spacy.load("en_core_web_sm")


class PrepositionalContext(BaseModel):
    target_token: str
    preposition: str
    governing_head: str
    head_pos: str
    attachment_type: str


def extract_pp_context(sentence: str, target_lemma: str) -> Optional[PrepositionalContext]:
    """
    Parses a sentence to find the target word (or its base lemma) and extracts its prepositional context.
    """
    doc = nlp(sentence)
    target_clean = target_lemma.lower().strip()
    
    for token in doc:
        # Match against either raw token text or base lemma
        if token.lemma_.lower() == target_clean or token.text.lower() == target_clean:
            
            # Check if target is object of preposition (pobj)
            if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                preposition = token.head
                governing_head = preposition.head
                
                attachment_type = "unknown"
                if governing_head.pos_ == "VERB":
                    attachment_type = "verb_attachment"
                elif governing_head.pos_ in ["NOUN", "PROPN"]:
                    attachment_type = "noun_modifier"
                    
                return PrepositionalContext(
                    target_token=token.text,
                    preposition=preposition.text,
                    governing_head=governing_head.text,
                    head_pos=governing_head.pos_,
                    attachment_type=attachment_type,
                )
                
    return None
