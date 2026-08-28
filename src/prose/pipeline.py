"""
PROSE Top-Level Pipeline Interface
Orchestrates candidate extraction, syntactic dependency parsing, and multi-stage filtering.
"""

from typing import Optional
from prose.core.lexical import extract_candidate_senses
from prose.core.parser import DependencyParser
from prose.core.filter import filter_candidate_senses, FilterResult


class ProsePipeline:
    def __init__(self, parser: Optional[DependencyParser] = None):
        """
        Initializes the pipeline. Accepts an injected DependencyParser to 
        prevent reloading the spaCy model during batch evaluations.
        """
        self.parser = parser if parser else DependencyParser()

    def process(self, sentence: str, target_token: str, pos: str = "NOUN") -> FilterResult:
        """
        Executes the PROSE disambiguation filter on a single target token within a sentence.
        """
        # 1. Parse the string into a spaCy Doc
        doc = self.parser.nlp(sentence)
        
        # 2. Extract strictly POS-restricted candidate senses (enforces "NOUN" to eliminate cross-POS inflation)
        candidates = extract_candidate_senses(target_token, pos=pos)
        
        # 3. Extract structural dependencies
        relations = self.parser.extract_relations(doc, target_token)
        ctx = relations[0] if relations else None
        
        # 4. Filter the candidates based on structure and constraints
        result = filter_candidate_senses(candidates, ctx, target_token)
        
        return result


def prose_filter(sentence: str, target_token: str, pos: str = "NOUN") -> FilterResult:
    """Convenience functional wrapper for single-shot pipeline execution."""
    pipeline = ProsePipeline()
    return pipeline.process(sentence, target_token, pos)