"""First-order-equivalent C score adapter; leave the frozen reference untouched."""
import torch

from features.ffn_target_consequence import SCORES
from features.ffn_visual_path_attribution import batched_scalar_gradients


def joint_score_adapter(callback):
    """Batch the three expensive suffix VJPs, then reuse the reference G VJPs.

    At EACH quadrature point, evaluate the true suffix and its three gradients.
    The returned graph has exactly those values and first derivatives at the
    current FFN output. It is not a clean-endpoint gradient approximation and
    does not promise higher-order derivatives.
    """
    def scores(output):
        if not torch.is_grad_enabled() or not output.requires_grad:
            return callback(output)
        if output.dtype not in (torch.float32, torch.float64):
            raise TypeError('Joint C VJP requires FP32/FP64')
        leaf=output.detach().requires_grad_(True)
        values=callback(leaf)
        if values.shape!=(*output.shape[:-1],len(SCORES)):
            raise ValueError('Three independent per-node scores required')
        gradients=batched_scalar_gradients({name:values[...,i] for i,name in enumerate(SCORES)},leaf)
        displacement=output-output.detach()
        tangent=torch.stack([(displacement*gradients[name]).sum(-1) for name in SCORES],dim=-1)
        return values.detach()+tangent
    return scores
