"""
PROSE Syntactic Parser Layer
"""
import spacy
from dataclasses import dataclass
from typing import Optional, List
from spacy.tokens import Doc, Token

@dataclass
class DependencyRelation:
    target_idx: int
    target_text: str
    target_lemma: str
    target_dep: str
    prep_idx: int
    prep_text: str
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
    def __init__(self, nlp: Optional[spacy.Language] = None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def _find_subject(self, governor: Token) -> Optional[Token]:
        """
        Traverse the syntax tree to locate the main clause subject.
        """
        for child in governor.children:
            if child.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
                return child
                
        if governor.dep_ in {"xcomp", "ccomp", "advcl"} and governor.head:
            for child in governor.head.children:
                if child.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
                    return child
                    
        return None

    def _is_passive(self, governor: Token) -> bool:
        """
        Robustly detect passive constructions beyond simple 'auxpass'.
        Catches "was victimized", "became victimized", and participial modifiers.
        """
        if any("pass" in child.dep_ for child in governor.children):
            return True
        if governor.head and governor.head.lemma_ in {"become", "get", "be"}:
            return True
        if governor.dep_ in {"acl", "amod"} and governor.text.endswith("ed"):
            return True
        return False

    def extract_relations(self, doc: Doc, target_word: str) -> List[DependencyRelation]:
        relations = []
        target_clean = target_word.lower().replace("_", " ")

        for token in doc:
            if token.text.lower() in target_clean or token.lemma_.lower() in target_clean:
                if token.dep_ == "pobj" and token.head.pos_ == "ADP":
                    prep_token = token.head
                    governor = prep_token.head
                    subject_token = self._find_subject(governor)
                    
                    rel = DependencyRelation(
                        target_idx=token.i,
                        target_text=token.text,
                        target_lemma=token.lemma_,
                        target_dep=token.dep_,
                        prep_idx=prep_token.i,
                        prep_text=prep_token.text,
                        prep_dep=prep_token.dep_,
                        governor_idx=governor.i,
                        governor_text=governor.text,
                        governor_lemma=governor.lemma_,
                        governor_pos=governor.pos_,
                        governor_dep=governor.dep_,
                        subject_text=subject_token.text if subject_token else None,
                        subject_lemma=subject_token.lemma_ if subject_token else None,
                        subject_pos=subject_token.pos_ if subject_token else None,
                        is_passive_governor=self._is_passive(governor)
                    )
                    relations.append(rel)
                    
        return relations