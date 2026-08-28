"""
PROSE Multi-Stage Candidate Elimination Engine
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .lexical import CandidateSense
from .parser import DependencyRelation

# ==========================================
# 1. DATA MODELS & ACTION LOGGING
# ==========================================

@dataclass
class ConstraintDecision:
    rule: str
    passed: bool
    eliminated_senses: List[str]
    reason: str

@dataclass
class FilterResult:
    target_word: str
    original_senses: List[str]
    surviving_senses: List[str]
    stage_applied: str
    csrr: float
    decisions: List[ConstraintDecision] = field(default_factory=list)


# ==========================================
# 2. STATIC PREPOSITION MAP & TAXONOMY
# ==========================================

STATIC_PREP_MAP = {
    "during": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.state", "noun.phenomenon"},
    "until": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "till": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "since": {"noun.time", "noun.event", "noun.act", "noun.state", "noun.process"},
    "after": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.phenomenon"},
    "before": {"noun.time", "noun.event", "noun.act", "noun.process", "noun.location", "noun.person", "noun.group", "noun.artifact"},
    "in": {"noun.location", "noun.group", "noun.time", "noun.state", "noun.cognition", "noun.phenomenon", "noun.artifact", "noun.substance"},
    "on": {"noun.location", "noun.time", "noun.communication", "noun.artifact", "noun.cognition"},
    "at": {"noun.location", "noun.time", "noun.event", "noun.act", "noun.group", "noun.artifact"},
    "from": {"noun.location", "noun.time", "noun.person", "noun.group", "noun.artifact", "noun.substance", "noun.cognition"},
    "over": {"noun.location", "noun.time", "noun.artifact", "noun.communication"},
    "under": {"noun.location", "noun.state", "noun.artifact", "noun.person", "noun.group"},
    "into": {"noun.location", "noun.state", "noun.artifact", "noun.substance", "noun.group"},
    "without": {"noun.artifact", "noun.substance", "noun.person", "noun.group", "noun.attribute", "noun.feeling"},
    "among": {"noun.person", "noun.group", "noun.animal", "noun.artifact", "noun.plant"},
    "out of": {"noun.location", "noun.state", "noun.artifact", "noun.substance", "noun.group", "noun.feeling"}
}

# --- ARCHITECT'S TAXONOMY PATCH (FINAL V0.3.1) ---
if "in" in STATIC_PREP_MAP:
    STATIC_PREP_MAP["in"].update({"noun.attribute", "noun.act", "noun.quantity", "noun.relation", "noun.communication", "noun.event", "noun.feeling", "noun.state", "noun.process"})
if "on" in STATIC_PREP_MAP:
    STATIC_PREP_MAP["on"].update({"noun.substance", "noun.communication", "noun.body", "noun.quantity", "noun.feeling", "noun.relation", "noun.state"})
if "at" in STATIC_PREP_MAP:
    STATIC_PREP_MAP["at"].update({"noun.plant", "noun.attribute", "noun.quantity", "noun.communication", "noun.object", "noun.animal", "noun.state", "noun.possession"})
if "from" in STATIC_PREP_MAP:
    STATIC_PREP_MAP["from"].update({"noun.attribute", "noun.quantity", "noun.communication", "noun.animal", "noun.cognition", "noun.feeling", "noun.phenomenon"})
if "into" in STATIC_PREP_MAP:
    STATIC_PREP_MAP["into"].update({"noun.communication", "noun.cognition"})


# Expanded affordance domains based on the Triage Protocol
INSTRUMENT_DOMAINS = {
    "noun.artifact", "noun.substance", "noun.communication", 
    "noun.act", "noun.attribute", "noun.cognition", "noun.relation",
    "noun.feeling", "noun.state", "noun.phenomenon", "noun.plant"
}

ATTRIBUTE_DOMAINS = {
    "noun.attribute", "noun.body", "noun.shape", "noun.state", "noun.relation", 
    "noun.person", "noun.animal", "noun.group", "noun.artifact",
    "noun.act", "noun.cognition", "noun.communication", "noun.substance",
    "noun.feeling", "noun.phenomenon", "noun.plant"
}

COMITATIVE_DOMAINS = {
    "noun.person", "noun.group", "noun.animal"
}

PASSIVE_AGENT_DOMAINS = {
    "noun.person", "noun.group", "noun.animal", "noun.act", "noun.event", 
    "noun.cognition", "noun.artifact", "noun.substance", "noun.phenomenon", 
    "noun.feeling", "noun.state"
}

MEANS_PROXIMITY_DOMAINS = {
    "noun.act", "noun.artifact", "noun.communication", "noun.location", 
    "noun.cognition", "noun.phenomenon", "noun.time", "noun.group", "noun.shape",
    "noun.relation", "noun.event"
}

# ==========================================
# 3. CORE FILTERING LOGIC
# ==========================================

def is_animate(ctx: DependencyRelation) -> bool:
    """
    Evaluates subject animacy using deterministic morphosyntactic markers 
    to prevent false eliminations of pronouns and anaphora.
    """
    if not ctx.subject_pos:
        return False
    if ctx.subject_pos in {"PRON", "PROPN"}:
        return True
    return False

def filter_candidate_senses(
    candidates: List[CandidateSense], 
    ctx: Optional[DependencyRelation], 
    target_word: str
) -> FilterResult:
    
    original_senses = [c.synset_id for c in candidates]
    surviving = original_senses.copy()
    stage_applied = "Stage 3 (Fallback - Preserved All)"
    decisions = []

    if ctx and candidates:
        prep_lemma = ctx.prep_lemma.lower()
        allowed_domains = set()
        stage_triggered = False
        rule_name = ""
        reason = ""

        if prep_lemma == "with":
            if ctx.governor_pos in {"VERB", "AUX"}:
                if is_animate(ctx):
                    allowed_domains = INSTRUMENT_DOMAINS | COMITATIVE_DOMAINS
                    reason = "Animate subject detected; allowing Comitative and Instrument."
                else:
                    allowed_domains = INSTRUMENT_DOMAINS
                    reason = "Inanimate subject detected; restricting to Instrument."
                rule_name = "Stage 2 (Comitative/Instrument Affordance)"
            else:
                allowed_domains = ATTRIBUTE_DOMAINS
                rule_name = "Stage 2 (Attribute/Companion Affordance)"
                reason = "Nominal governor detected; allowing Attribute/Companion domains."
            stage_triggered = True

        elif prep_lemma == "by":
            if ctx.is_passive_governor:
                allowed_domains = PASSIVE_AGENT_DOMAINS
                rule_name = "Stage 2 (Passive Agent Affordance)"
                reason = "Passive voice detected; allowing extended Agent domains."
            elif ctx.governor_pos in {"VERB", "AUX"}:
                allowed_domains = MEANS_PROXIMITY_DOMAINS
                rule_name = "Stage 2 (Means/Proximity Affordance)"
                reason = "Active voice verbal governor detected; allowing Means/Proximity."
            stage_triggered = True

        elif prep_lemma in STATIC_PREP_MAP:
            allowed_domains = STATIC_PREP_MAP[prep_lemma]
            rule_name = "Stage 1 (Direct Constraint)"
            reason = f"Static mapping applied for preposition '{prep_lemma}'."
            stage_triggered = True

        if stage_triggered:
            # Globally protect foundational concepts and food from broad static elimination
            allowed_domains.update({"noun.Tops", "noun.food"})
            filtered = [c.synset_id for c in candidates if c.lexname in allowed_domains]
            eliminated = list(set(original_senses) - set(filtered))
            
            if filtered:
                surviving = filtered
                stage_applied = rule_name
                decisions.append(ConstraintDecision(
                    rule=rule_name,
                    passed=True,
                    eliminated_senses=eliminated,
                    reason=reason
                ))
            else:
                stage_applied = "Stage 3 (Fallback - Empty Intersect Guard)"
                decisions.append(ConstraintDecision(
                    rule=rule_name,
                    passed=False,
                    eliminated_senses=[],
                    reason="Empty intersect detected. Reverting to fail-open safety."
                ))

    csrr = 1.0 - (len(surviving) / len(original_senses)) if original_senses else 0.0

    return FilterResult(
        target_word=target_word,
        original_senses=original_senses,
        surviving_senses=surviving,
        stage_applied=stage_applied,
        csrr=csrr,
        decisions=decisions
    )