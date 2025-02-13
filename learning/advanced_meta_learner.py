# learning/advanced_meta_learner.py
import torch
from higher import innerloop_ctx

class AdvancedMetaLearner:
    """
    A MAML-style meta-learner that adapts both weights and (in future) structure.
    """
    def __init__(self, base_model, lr_outer=1e-4, lr_inner=1e-3):
        self.base_model = base_model
        self.lr_outer = lr_outer
        self.lr_inner = lr_inner
        self.outer_opt = torch.optim.Adam(self.base_model.parameters(), lr=self.lr_outer)

    def meta_train_step(self, support_data, query_data, loss_fn, steps=5):
        x_sup, y_sup = support_data
        x_qry, y_qry = query_data
        with innerloop_ctx(self.base_model, self.outer_opt) as (fmodel, diffopt):
            for _ in range(steps):
                preds = fmodel(x_sup)
                loss = loss_fn(preds, y_sup)
                diffopt.step(loss)
            preds_q = fmodel(x_qry)
            q_loss = loss_fn(preds_q, y_qry)
            self.outer_opt.zero_grad()
            q_loss.backward()
            self.outer_opt.step()
        return q_loss.item()

    def spawn_specialized_model(self):
        import copy
        return copy.deepcopy(self.base_model)

