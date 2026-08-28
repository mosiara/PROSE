from prose.pipeline import prose_filter

def test_pipeline():
    # Testing the core pipeline without any sys.path hacks
    sentence = "She traveled with a companion."
    target = "companion"
    
    print(f"Analyzing: '{sentence}' (Target: {target})")
    result = prose_filter(sentence, target)
    
    print(f"Stage Applied: {result.stage_applied}")
    print(f"Original Senses: {len(result.original_senses)}")
    print(f"Surviving Senses: {len(result.surviving_senses)}")
    print(f"CSRR: {result.csrr:.1%}")

if __name__ == "__main__":
    test_pipeline()