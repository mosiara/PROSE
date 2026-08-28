import pytest
from prose.pipeline import prose_filter

# Structured as: (sentence, target_word, required_gold_sense)
FALSE_ELIMINATION_CASES = [
    ("He got hit from the blind side by the split end coming back on the second play of the game .", "side", "side.n.03"),
    ("He just shot at the board and then drew circles around the holes to form a bull's-eye .", "board", "board.n.02"),
    ("Besides , he can hardly avoid musing on the instability of death...", "instability", "instability.n.02"),
    ("Also on the bill at the Fifty-fifth Street is a nice ten minute color film...", "bill", "bill.n.04"),
    ("This comes not alone from high-set , high-rep training...", "training", "training.n.01"),
    ("Look at the physical features of the land to determine how desirable it is...", "features", "feature.n.01"),
    ("The suggested course of the A.I.D. was based on the usual course offered...", "course", "course.n.01"),
    ("At the end of its letter was the information...", "end", "end.n.05"),
    ("On the one side we have the university professors and their students...", "side", "side.n.02"),
    ("Dominated by the vicious circle of the university promotion system...", "vicious_circle", "vicious_circle.n.01")
]

@pytest.mark.parametrize("sentence, target, gold_sense", FALSE_ELIMINATION_CASES)
def test_false_eliminations(sentence, target, gold_sense):
    result = prose_filter(sentence, target, pos="NOUN")
    decision_log = result.decisions[0] if result.decisions else None
    
    assert gold_sense in result.surviving_senses, (
        f"\n[REGRESSION FAILED] Gold sense '{gold_sense}' falsely eliminated from '{target}'.\n"
        f"Stage Applied: {result.stage_applied}\n"
        f"Reasoning: {decision_log.reason if decision_log else 'N/A'}\n"
        f"Eliminated Senses: {decision_log.eliminated_senses if decision_log else []}"
    )