"""
PROSE Multi-Stage Candidate Elimination Engine
"""
from dataclasses import dataclass
from typing import List, Optional
from spacy.tokens import Doc

# I FIXED THIS LINE TO MATCH YOUR ACTUAL FOLDERS:
from core.lexical import extract_candidate_senses as get_wordnet_candidates, CandidateSense as WordNetCandidate

@dataclass
class FilterResult:
    target_word: str
    original_senses: List[str]
    surviving_senses: List[str]
    stage_applied: str
    csrr: float


# Stage 1: Deterministic Preposition -> Lexicographer Domain Constraints
STATIC_PREP_MAP = {
    # ---------------------------------------------------------
    # 1. STRICT TEMPORAL 
    # ---------------------------------------------------------
    "during": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.state", "noun.phenomenon"},
    "until": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "till": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "since": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "after": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.phenomenon"},
    "before": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.location", "noun.person", "noun.group", "noun.artifact"},
    "pending": {"noun.event", "noun.act", "noun.communication", "noun.cognition", "noun.process"},
    
    # ---------------------------------------------------------
    # 2. STRICT LOCATIVE & CONTAINMENT 
    # ---------------------------------------------------------
    "inside": {"noun.location", "noun.artifact", "noun.object", "noun.body", "noun.group", "noun.animal", "noun.plant", "noun.shape"},
    "outside": {"noun.location", "noun.artifact", "noun.object", "noun.group", "noun.animal", "noun.body"},
    "within": {"noun.location", "noun.artifact", "noun.group", "noun.time", "noun.state", "noun.quantity", "noun.relation", "noun.cognition", "noun.communication", "noun.act"},
    "aboard": {"noun.artifact", "noun.location", "noun.object"}, 
    
    # ---------------------------------------------------------
    # 3. BROAD LOCATIVE / ABSTRACT (Highly Flexible)
    # ---------------------------------------------------------
    "in": {
        "noun.location", "noun.artifact", "noun.group", "noun.state", 
        "noun.event", "noun.time", "noun.act", "noun.cognition", 
        "noun.communication", "noun.body", "noun.phenomenon", "noun.substance", 
        "noun.feeling", "noun.possession", "noun.shape", "noun.process", "noun.object"
    },
    "on": {
        "noun.location", "noun.artifact", "noun.time", "noun.event", 
        "noun.communication", "noun.cognition", "noun.body", "noun.state",
        "noun.object", "noun.animal", "noun.plant", "noun.shape", "noun.act"
    },
    "at": {
        "noun.location", "noun.artifact", "noun.time", "noun.event", 
        "noun.group", "noun.state", "noun.quantity", "noun.object", 
        "noun.person", "noun.animal", "noun.act", "noun.communication", 
        "noun.cognition", "noun.feeling", "noun.phenomenon"
    },

    # ---------------------------------------------------------
    # 4. PATH, DIRECTION & MOVEMENT
    # ---------------------------------------------------------
    "into": {
        "noun.location", "noun.artifact", "noun.state", "noun.event", 
        "noun.act", "noun.cognition", "noun.group", "noun.substance", 
        "noun.object", "noun.body", "noun.animal", "noun.process", "noun.shape"
    },
    "onto": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.animal"},
    "toward": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.event", "noun.time", "noun.state", "noun.object", "noun.animal", "noun.cognition", "noun.act"},
    "towards": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.event", "noun.time", "noun.state", "noun.object", "noun.animal", "noun.cognition", "noun.act"},
    "through": {"noun.location", "noun.artifact", "noun.process", "noun.event", "noun.act", "noun.time", "noun.state", "noun.communication", "noun.object", "noun.body", "noun.substance", "noun.group", "noun.cognition"},
    "across": {"noun.location", "noun.artifact", "noun.body", "noun.group", "noun.object", "noun.time", "noun.event", "noun.communication"},
    "past": {"noun.location", "noun.artifact", "noun.person", "noun.time", "noun.event", "noun.object", "noun.group"},
    "along": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.group", "noun.communication"},

    # ---------------------------------------------------------
    # 5. SOURCE & SEPARATION
    # ---------------------------------------------------------
    "from": {
        "noun.location", "noun.person", "noun.group", "noun.time", 
        "noun.event", "noun.artifact", "noun.state", "noun.cognition", 
        "noun.communication", "noun.relation", "noun.attribute",
        "noun.object", "noun.shape", "noun.body", "noun.animal", 
        "noun.plant", "noun.phenomenon", "noun.act", "noun.process", 
        "noun.substance", "noun.possession", "noun.motive"
    },
    "off": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.time", "noun.state", "noun.communication"},
    "out of": {"noun.location", "noun.artifact", "noun.state", "noun.group", "noun.substance", "noun.feeling", "noun.object", "noun.body", "noun.act", "noun.cognition", "noun.time"},

    # ---------------------------------------------------------
    # 6. VERTICALITY & HIERARCHY 
    # ---------------------------------------------------------
    "above": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.communication", "noun.object", "noun.animal", "noun.relation"},
    "below": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.object", "noun.animal", "noun.relation"},
    "over": {"noun.location", "noun.artifact", "noun.time", "noun.event", "noun.person", "noun.group", "noun.quantity", "noun.state", "noun.communication", "noun.object", "noun.body", "noun.animal", "noun.process", "noun.act"},
    "under": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.state", "noun.cognition", "noun.communication", "noun.object", "noun.body", "noun.animal", "noun.process", "noun.act"},
    "beneath": {"noun.location", "noun.artifact", "noun.state", "noun.person", "noun.object", "noun.body", "noun.animal", "noun.group", "noun.cognition"},
    "underneath": {"noun.location", "noun.artifact", "noun.body", "noun.object", "noun.animal", "noun.person"},

    # ---------------------------------------------------------
    # 7. PROXIMITY & ALIGNMENT
    # ---------------------------------------------------------
    "near": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.time", "noun.event", "noun.object", "noun.animal", "noun.state"},
    "against": {"noun.artifact", "noun.person", "noun.group", "noun.location", "noun.cognition", "noun.communication", "noun.act", "noun.state", "noun.object", "noun.animal", "noun.body", "noun.phenomenon"},
    "alongside": {"noun.location", "noun.artifact", "noun.person", "noun.group", "noun.object", "noun.animal", "noun.event"},
    "amid": {"noun.group", "noun.event", "noun.state", "noun.act", "noun.phenomenon", "noun.location", "noun.artifact", "noun.feeling"},
    "among": {"noun.group", "noun.person", "noun.artifact", "noun.animal", "noun.plant", "noun.location", "noun.object"},
    "between": {"noun.group", "noun.person", "noun.artifact", "noun.location", "noun.time", "noun.event", "noun.state", "noun.relation", "noun.object", "noun.animal", "noun.cognition", "noun.act"},

    # ---------------------------------------------------------
    # 8. AGENCY, INSTRUMENT & ASSOCIATION
    # ---------------------------------------------------------
    "by": {
        "noun.person", "noun.group", "noun.artifact", "noun.act", 
        "noun.event", "noun.time", "noun.location", "noun.communication", 
        "noun.cognition", "noun.animal", "noun.phenomenon", "noun.process", 
        "noun.state", "noun.object", "noun.quantity", "noun.relation"
    },
    "with": {
        "noun.person", "noun.group", "noun.artifact", "noun.attribute", 
        "noun.feeling", "noun.state", "noun.substance", "noun.animal", 
        "noun.possession", "noun.communication", "noun.act", "noun.event", 
        "noun.cognition", "noun.object", "noun.plant", "noun.phenomenon", "noun.process"
    },
    "without": {
        "noun.person", "noun.group", "noun.artifact", "noun.attribute", 
        "noun.feeling", "noun.state", "noun.substance", "noun.possession", 
        "noun.animal", "noun.object", "noun.cognition", "noun.communication", "noun.act", "noun.event"
    },
    "via": {"noun.artifact", "noun.location", "noun.person", "noun.communication", "noun.act", "noun.group", "noun.process", "noun.cognition"}
}

# [✨ Creative Note: Affordance Dictionaries for Stage 2 Resolution]
INSTRUMENT_DOMAINS = {"noun.artifact", "noun.substance", "noun.communication"}
ATTRIBUTE_DOMAINS = {"noun.attribute", "noun.body", "noun.shape", "noun.state", "noun.relation"}
# Expanded to include abstract victimizers (e.g., 'a form of athletics', diseases, laws)
AGENT_DOMAINS = {
    "noun.person", 
    "noun.group", 
    "noun.animal",
    "noun.act",        # Actions/athletics that can 'victimize'
    "noun.event",      # Events causing an outcome
    "noun.cognition",  # Abstract rules/laws
    "noun.state"       # Conditions/diseases
}
TRANSPORT_DOMAINS = {"noun.artifact"}


def find_target_token(doc: Doc, target_word: str):
    """Locate the target token in the parsed document."""
    for token in doc:
        if token.text.lower() == target_word.lower() or token.lemma_.lower() == target_word.lower():
            return token
    return None


def run_multistage_filter(doc: Doc, target_word: str) -> FilterResult:
    """
    Executes the 3-Stage PROSE pruning pipeline.
    """
    candidates = get_wordnet_candidates(target_word)
    original_sense_keys = [c.synset_id for c in candidates]

    if not candidates:
        return FilterResult(target_word, [], [], "Stage 0 (No Candidates)", 0.0)

    token = find_target_token(doc, target_word)
    if token is None:
        return FilterResult(target_word, original_sense_keys, original_sense_keys, "Stage 3 (Fallback - Unresolved)", 0.0)

    surviving: List[WordNetCandidate] = []
    stage_applied = "Stage 3 (Fallback - Preserved All)"

    # Check if target is the object of a preposition
    if token.dep_ == "pobj" and token.head.pos_ == "ADP":
        prep_token = token.head
        prep_lemma = prep_token.lemma_.lower()
        governor = prep_token.head

        # STAGE 1: Direct Mapping for unambiguous prepositions
        if prep_lemma in STATIC_PREP_MAP:
            allowed_domains = STATIC_PREP_MAP[prep_lemma]
            surviving = [c for c in candidates if c.lexname in allowed_domains]
            if surviving:
                stage_applied = "Stage 1 (Direct Constraint)"

        # STAGE 2: Syntactic Attachment + Affordance Evaluation
        elif prep_lemma == "with":
            if governor.pos_ == "VERB":
                surviving = [c for c in candidates if c.lexname in INSTRUMENT_DOMAINS]
                if surviving:
                    stage_applied = "Stage 2 (Instrument Affordance)"
            elif governor.pos_ in {"NOUN", "PROPN"}:
                surviving = [c for c in candidates if c.lexname in ATTRIBUTE_DOMAINS]
                if surviving:
                    stage_applied = "Stage 2 (Attribute Affordance)"

        elif prep_lemma == "by":
            # [✨ Creative Note: Passive Agency vs. Means vs. Proximity Heuristic]
            is_passive = any(child.dep_ == "auxpass" for child in governor.children)
            if is_passive:
                surviving = [c for c in candidates if c.lexname in AGENT_DOMAINS]
                if surviving:
                    stage_applied = "Stage 2 (Passive Agent Affordance)"
            elif governor.pos_ == "VERB":
                surviving = [c for c in candidates if c.lexname in (TRANSPORT_DOMAINS | {"noun.location"})]
                if surviving:
                    stage_applied = "Stage 2 (Means/Proximity Affordance)"

    # STAGE 3: Conservative Fallback Guard
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