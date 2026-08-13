from src.recall_aware.abstention import AbstentionPolicy

policy = AbstentionPolicy(threshold=0.75)

tests = [0.20, 0.50, 0.74, 0.75, 0.90]

for score in tests:

    result = policy.decide(score)

    print(
        f"{score:.2f} -> "
        f"{result.decision}"
    )