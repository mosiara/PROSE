from prose.core.parser import extract_pp_context
from prose.core.lexical import extract_candidate_senses
from prose.core.filter import run_multistage_filter

class ProsePipeline:
    def __init__(self):
        pass

    def process(self, sentence: str, target_word: str):
        # Run the full integrated pipeline
        return run_multistage_filter(sentence, target_word)
