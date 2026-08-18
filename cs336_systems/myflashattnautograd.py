import torch
import math

class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):
        d = k.size(-1)
        s = q @ k.transpose(-2, -1) / math.sqrt(d)
        if is_causal:
            n_queries = q.size(-2)
            n_keys = k.size(-2)
            causal_mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
            s = torch.where(causal_mask, s, -1e6)
        L = torch.logsumexp(s, dim=-1)
        ctx.save_for_backward(q, k, v, L)
        ctx.is_causal = is_causal
        P = torch.exp(s - L.unsqueeze(-1))
        return P @ v

    @staticmethod
    def backward(ctx, o_grad):
        q, k, v, L = ctx.saved_tensors
        d = k.size(-1)
        s = q @ k.transpose(-2, -1) / math.sqrt(d)
        if ctx.is_causal:
            n_queries = q.size(-2)
            n_keys = k.size(-2)
            causal_mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
            s = torch.where(causal_mask, s, -1e6)
        P = torch.exp(s - L.unsqueeze(-1))
        O = P @ v
        D = (O * o_grad).sum(dim=-1)
        dv = P.transpose(-2, -1) @ o_grad
        p_grad = o_grad @ v.transpose(-2, -1)
        s_grad = P * (p_grad - D.unsqueeze(-1))
        dq = s_grad @ k / math.sqrt(d)
        dk = s_grad.transpose(-2, -1) @ q / math.sqrt(d)
        return dq, dk, dv, None
