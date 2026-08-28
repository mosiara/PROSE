"""
PROSE Lexical Extraction Layer
Retrieves candidate WordNet synsets and semantic domain metadata for a target token.
"""

from typing import List, Optional
from pydantic import BaseModel
from nltk.corpus import wordnet as wn


class CandidateSense(BaseModel):
    synset_id: str
    lemmas: List[str]
    pos: str
    definition: str
    examples: List[str]
    lexname: str  # Lexicographer file name, e.g., 'noun.artifact', 'noun.act', 'noun.location'


def get_wordnet_pos(spacy_pos: str) -> Optional[str]:
    """Maps spaCy POS tags to WordNet POS tags."""
    pos_map = {
        "NOUN": wn.NOUN,
        "VERB": wn.VERB,
        "ADJ": wn.ADJ,
        "ADV": wn.ADV,
    }
    return pos_map.get(spacy_pos.upper())


def extract_candidate_senses(lemma: str, pos: Optional[str] = None) -> List[CandidateSense]:
    """
    Pulls all candidate synsets for a target lemma from WordNet.
    
    Args:
        lemma: The base form of the target word (e.g., 'net', 'bank', 'spot').
        pos: Optional spaCy POS string ('NOUN', 'VERB', etc.) to pre-filter by part of speech.
    
    Returns:
        A list of CandidateSense objects representing the baseline pool before filtering.
    """
    wn_pos = get_wordnet_pos(pos) if pos else None
    synsets = wn.synsets(lemma, pos=wn_pos)
    
    candidates = []
    for s in synsets:
        candidates.append(
            CandidateSense(
                synset_id=s.name(),
                lemmas=[l.name() for l in s.lemmas()],
                pos=s.pos(),
                definition=s.definition(),
                examples=s.examples(),
                lexname=s.lexname(),
            )
        )
    return candidates
