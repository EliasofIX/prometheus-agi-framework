# learning/learning_strategy_optimizer.py
class LearningStrategyOptimizer:
    """
    Dynamically adjusts meta-learning hyperparameters based on performance.
    """
    def __init__(self, meta_learner):
        self.meta_learner = meta_learner
        self.performance_log = []

    def update_strategy(self, recent_loss):
        self.performance_log.append(recent_loss)
        if len(self.performance_log) >= 5:
            avg_loss = sum(self.performance_log[-5:]) / 5.0
            if avg_loss > 1.0:
                for group in self.meta_learner.outer_opt.param_groups:
                    group['lr'] *= 1.1

