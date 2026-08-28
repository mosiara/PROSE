"""
PROSE Top-Level Pipeline Interface
Orchestrates candidate extraction, syntactic dependency parsing, and multi-stage filtering.
"""

from typing import Optional, Dict, Any
from src.prose.core.lexical import extract_candidate_senses
from src.prose.core.parser import extract_pp_context
from src.prose.core.filter import filter_candidate_senses, FilterResult


class ProsePipeline:
    def __init__(self):
        pass

    def run(self, sentence: str, target_token: str, pos: Optional[str] = "NOUN") -> FilterResult:
        """
        Executes the PROSE disambiguation filter on a single target token within a sentence.
        """
        # Step 1: Retrieve all candidate senses
        candidates = extract_candidate_senses(target_token, pos=pos)
        
        # Step 2: Extract prepositional attachment context
        pp_ctx = extract_pp_context(sentence, target_token)
        
        # Step 3: Run multi-stage filter
        result = filter_candidate_senses(candidates, pp_ctx)
        return result


def prose_filter(sentence: str, target_token: str, pos: Optional[str] = "NOUN") -> FilterResult:
    """Convenience functional wrapper for the pipeline."""
    pipeline = ProsePipeline()
    return pipeline.run(sentence, target_token, pos)
