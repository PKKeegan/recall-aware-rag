import numpy as np


class RecallEstimator:

    """
    Estimates retrieval recall from
    similarity scores.
    """

    def estimate(self, nodes):

        if len(nodes) == 0:
            return 0.0

        scores = []

        for node in nodes:

            if node.score is None:
                continue

            scores.append(node.score)

        if len(scores) == 0:
            return 0.0

        scores = np.array(scores)

        scores = (
            scores - scores.min()
        ) / (
            scores.max() - scores.min() + 1e-8
        )

        return float(scores.mean())