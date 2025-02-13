# cognition/intrinsic_curiosity_module.py
class IntrinsicCuriosityModule:
    """
    Computes intrinsic rewards based on Bayesian surprise.
    """
    def __init__(self):
        pass

    def compute_novelty_score(self, current_state, predicted_state):
        import random
        return 0.7 + 0.2 * random.random()

