from dataclasses import dataclass
from typing import List, Optional
import spacy
from spacy.tokens import Doc, Token

@dataclass
class DependencyRelation:
    target_idx: int
    target_text: str
    target_lemma: str
    target_dep: str
    prep_idx: int
    prep_text: str
    prep_lemma: str
    prep_dep: str
    governor_idx: int
    governor_text: str
    governor_lemma: str
    governor_pos: str
    governor_dep: str
    subject_text: Optional[str] = None
    subject_lemma: Optional[str] = None
    subject_pos: Optional[str] = None
    is_passive_governor: bool = False


class DependencyParser:
    def __init__(self, model: str = "en_core_web_sm"):
        """Initializes the parser with the specified spaCy model."""
        self.nlp = spacy.load(model)

    def _is_passive(self, governor: Token) -> bool:
        """
        Identifies if the governor verb is in a passive construction
        using strict morphological and dependency markers.
        """
        # Strict check for passive auxiliary
        if any(child.dep_ == "auxpass" for child in governor.children):
            return True
            
        # Check for participial modifier using exact VBN (past participle) tag
        if governor.dep_ in {"acl", "amod"} and governor.tag_ == "VBN":
            return True
            
        return False

    def _find_subject(self, governor: Token) -> Optional[Token]:
        """
        Traverses the syntax tree to locate the main clause subject.
        """
        for child in governor.children:
            if "subj" in child.dep_:
                return child
                
        if governor.dep_ in {"xcomp", "ccomp", "advcl"} and governor.head:
            for child in governor.head.children:
                if "subj" in child.dep_:
                    return child
                    
        return None

    def extract_relations(self, doc: Doc, target_word: str) -> List[DependencyRelation]:
        """
        Extracts structural relationships between the target word, its preposition,
        the governing verb, and the main clause subject.
        """
        relations = []
        # Exact token binding to prevent substring matches (e.g., 'art' matching 'cart')
        target_tokens = set(target_word.lower().replace("_", " ").split())

        for token in doc:
            if token.text.lower() in target_tokens or token.lemma_.lower() in target_tokens:
                if token.head.pos_ == "ADP":
                    prep = token.head
                    prep_text = prep.text
                    prep_lemma = prep.lemma_
                    governor = prep.head
                    
                    # Multi-word preposition normalization (e.g., "out of")
                    if prep.i > 0 and doc[prep.i - 1].lemma_.lower() == "out" and prep.lemma_.lower() == "of":
                        prep_text = "out of"
                        prep_lemma = "out of"
                        governor = doc[prep.i - 1].head

                    subject = self._find_subject(governor)

                    relations.append(DependencyRelation(
                        target_idx=token.i,
                        target_text=token.text,
                        target_lemma=token.lemma_,
                        target_dep=token.dep_,
                        prep_idx=prep.i,
                        prep_text=prep_text,
                        prep_lemma=prep_lemma,
                        prep_dep=prep.dep_,
                        governor_idx=governor.i,
                        governor_text=governor.text,
                        governor_lemma=governor.lemma_,
                        governor_pos=governor.pos_,
                        governor_dep=governor.dep_,
                        subject_text=subject.text if subject else None,
                        subject_lemma=subject.lemma_ if subject else None,
                        subject_pos=subject.pos_ if subject else None,
                        is_passive_governor=self._is_passive(governor)
                    ))
                    
        return relations