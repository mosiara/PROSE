"""
PROSE Multi-Stage Candidate Elimination Engine
"""
from dataclasses import dataclass
from typing import List, Optional
from spacy.tokens import Doc

from .lexical import extract_candidate_senses as get_wordnet_candidates, CandidateSense as WordNetCandidate
from .parser import DependencyParser, DependencyRelation

@dataclass
class FilterResult:
    target_word: str
    original_senses: List[str]
    surviving_senses: List[str]
    stage_applied: str
    csrr: float

STATIC_PREP_MAP = {
    "during": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.state", "noun.phenomenon"},
    "until": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "till": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "since": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "after": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.phenomenon"},
    "before": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.location", "noun.person", "noun.group", "noun.artifact"},
    "pending": {"noun.event", "noun.act", "noun.communication", "noun.cognition", "noun.process"},
    "inside": {"noun.location", "noun.artifact", "noun.object", "noun.body", "noun.group", "noun.animal", "noun.plant", "noun.shape"},
    "outside": {"noun.location", "noun.artifact", "noun.object", "noun.group", "noun.animal", "noun.body"},
    "within": {"noun.location", "noun.artifact", "noun.group", "noun.time", "noun.state", "noun.quantity", "noun.relation", "noun.cognition", "noun.communication", "noun.act"},
    "aboard": {"noun.artifact", "noun.location", "noun.object"}, 
    "in": {"noun.location", "noun.artifact", "noun.group", "noun.state", "noun.event", "noun.time", "noun.act", "noun.cognition", "noun.communication", "noun.body", "noun.phenomenon", "noun.substance", "noun.feeling", "noun.possession", "noun.shape", "noun.process", "noun.object", "noun.Tops"},
    "on": {"noun.location", "noun.artifact", "noun.time", "noun.event", "noun.communication", "noun.cognition", "noun.body", "noun.state", "noun.object", "noun.animal", "noun.plant", "noun.shape", "noun.act", "noun.Tops"},
    "at": {"noun.location", "noun.artifact", "noun.time", "noun.event", "noun.group", "noun.state", "noun.quantity", "noun.object", "noun.person", "noun.animal", "noun.act", "noun.communication", "noun.cognition", "noun.feeling", "noun.phenomenon", "noun.Tops"},
    "into": {"noun.location", "noun.artifact", "noun.state", "noun.event", "noun.act", "noun.cognition", "noun.group", "noun.substance", "noun.object", "noun.body", "noun.animal", "noun.process", "noun.shape"},
    "onto": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.animal"},
    "toward": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.event", "noun.time", "noun.state", "noun.object", "noun.animal", "noun.cognition", "noun.act"},
    "towards": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.event", "noun.time", "noun.state", "noun.object", "noun.animal", "noun.cognition", "noun.act"},
    "through": {"noun.location", "noun.artifact", "noun.process", "noun.event", "noun.act", "noun.time", "noun.state", "noun.communication", "noun.object", "noun.body", "noun.substance", "noun.group", "noun.cognition"},
    "across": {"noun.location", "noun.artifact", "noun.body", "noun.group", "noun.object", "noun.time", "noun.event", "noun.communication"},
    "past": {"noun.location", "noun.artifact", "noun.person", "noun.time", "noun.event", "noun.object", "noun.group"},
    "along": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.group", "noun.communication"},
    "from": {"noun.location", "noun.person", "noun.group", "noun.time", "noun.event", "noun.artifact", "noun.state", "noun.cognition", "noun.communication", "noun.relation", "noun.attribute", "noun.object", "noun.shape", "noun.body", "noun.animal", "noun.plant", "noun.phenomenon", "noun.act", "noun.process", "noun.substance", "noun.possession", "noun.motive"},
    "off": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.time", "noun.state", "noun.communication"},
    "out of": {"noun.location", "noun.artifact", "noun.state", "noun.group", "noun.substance", "noun.feeling", "noun.object", "noun.body", "noun.act", "noun.cognition", "noun.time"},
    "above": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.communication", "noun.object", "noun.animal", "noun.relation"},
    "below": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.object", "noun.animal", "noun.relation"},
    "over": {"noun.location", "noun.artifact", "noun.time", "noun.event", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.communication", "noun.object", "noun.body", "noun.animal", "noun.process", "noun.act"},
    "under": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.state", "noun.cognition", "noun.communication", "noun.object", "noun.body", "noun.animal", "noun.process", "noun.act"},
    "beneath": {"noun.location", "noun.artifact", "noun.state", "noun.person", "noun.object", "noun.body", "noun.animal", "noun.group", "noun.cognition"},
    "underneath": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.animal", "noun.person"},
    "near": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.time", "noun.event", "noun.object", "noun.animal", "noun.state"},
    "against": {"noun.artifact", "noun.person", "noun.group", "noun.location", "noun.cognition", "noun.communication", "noun.act", "noun.state", "noun.object", "noun.animal", "noun.body", "noun.phenomenon"},
    "alongside": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.object", "noun.animal", "noun.event"},
    "amid": {"noun.group", "noun.event", "noun.state", "noun.act", "noun.phenomenon", "noun.location", "noun.artifact", "noun.feeling"},
    "among": {"noun.group", "noun.person", "noun.artifact", "noun.animal", "noun.plant", "noun.location", "noun.object"},
    "between": {"noun.group", "noun.person", "noun.artifact", "noun.location", "noun.time", "noun.event", "noun.state", "noun.relation", "noun.object", "noun.animal", "noun.cognition", "noun.act"},
    "without": {"noun.person", "noun.group", "noun.artifact", "noun.attribute", "noun.feeling", "noun.state", "noun.substance", "noun.possession", "noun.animal", "noun.object", "noun.cognition", "noun.communication", "noun.act", "noun.event"},
    "via": {"noun.artifact", "noun.location", "noun.person", "noun.communication", "noun.act", "noun.group", "noun.process", "noun.cognition"}
}

INSTRUMENT_DOMAINS = {"noun.artifact", "noun.substance", "noun.communication"}
COMITATIVE_DOMAINS = {"noun.person", "noun.group", "noun.animal"}
ATTRIBUTE_DOMAINS = {"noun.attribute", "noun.body", "noun.shape", "noun.state", "noun.relation"}
AGENT_DOMAINS = {"noun.person", "noun.group", "noun.animal", "noun.act", "noun.event", "noun.cognition", "noun.state"}
TRANSPORT_DOMAINS = {"noun.artifact"}

def is_animate(lemma: str) -> bool:
    if not lemma:
        return False
    cands = get_wordnet_candidates(lemma)
    return any(c.lexname in COMITATIVE_DOMAINS for c in cands)

def filter_candidate_senses(candidates: List[WordNetCandidate], ctx: Optional[DependencyRelation], target_word: str) -> FilterResult:
    original_sense_keys = [c.synset_id for c in candidates]
    
    if not candidates:
        return FilterResult(target_word, [], [], "Stage 0 (No Candidates)", 0.0)
        
    if not ctx:
        return FilterResult(target_word, original_sense_keys, original_sense_keys, "Stage 3 (Fallback - Unresolved)", 0.0)

    surviving: List[WordNetCandidate] = []
    stage_applied = "Stage 3 (Fallback - Preserved All)"
    prep_lemma = ctx.prep_lemma.lower() if hasattr(ctx, 'prep_lemma') else ctx.prep_text.lower()

    if prep_lemma in STATIC_PREP_MAP:
        allowed_domains = STATIC_PREP_MAP[prep_lemma]
        surviving = [c for c in candidates if c.lexname in allowed_domains]
        if surviving:
            stage_applied = "Stage 1 (Direct Constraint)"

    elif prep_lemma == "with":
        if ctx.governor_pos == "VERB":
            # Unlock Comitative for Proper Nouns OR explicitly animate dictionary words
            if ctx.subject_pos == "PROPN" or (ctx.subject_lemma and is_animate(ctx.subject_lemma)):
                allowed = INSTRUMENT_DOMAINS | COMITATIVE_DOMAINS
                stage_applied = "Stage 2 (Comitative/Instrument Affordance)"
            else:
                allowed = INSTRUMENT_DOMAINS
                stage_applied = "Stage 2 (Instrument Affordance)"
                
            surviving = [c for c in candidates if c.lexname in allowed]
        elif ctx.governor_pos in {"NOUN", "PROPN"}:
            surviving = [c for c in candidates if c.lexname in ATTRIBUTE_DOMAINS]
            if surviving:
                stage_applied = "Stage 2 (Attribute Affordance)"

    elif prep_lemma == "by":
        if ctx.is_passive_governor or "pass" in ctx.governor_dep or "pass" in ctx.target_dep:
            surviving = [c for c in candidates if c.lexname in AGENT_DOMAINS]
            if surviving:
                stage_applied = "Stage 2 (Passive Agent Affordance)"
        elif ctx.governor_pos == "VERB":
            surviving = [c for c in candidates if c.lexname in (TRANSPORT_DOMAINS | {"noun.location"})]
            if surviving:
                stage_applied = "Stage 2 (Means/Proximity Affordance)"

    if not surviving:
        surviving = candidates
        stage_applied = "Stage 3 (Fallback - Preserved All)"

    surviving_keys = [c.synset_id for c in surviving]
    csrr = 1.0 - (len(surviving_keys) / len(original_sense_keys)) if original_sense_keys else 0.0

    return FilterResult(
        target_word=target_word,
        original_senses=original_sense_keys,
        surviving_senses=surviving_keys,
        stage_applied=stage_applied,
        csrr=csrr
    )

_parser_instance = None

def run_multistage_filter(doc: Doc, target_word: str) -> FilterResult:
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = DependencyParser()
        
    candidates = get_wordnet_candidates(target_word)
    relations = _parser_instance.extract_relations(doc, target_word)
    ctx = relations[0] if relations else None
         
    return filter_candidate_senses(candidates, ctx, target_word)