import torch
import math
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,  
    IS_CAUSAL: tl.constexpr, 
):

    """
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

    Q shape = (batch_size, n_queries, d_head)
    K shape = (batch_size, n_keys, d_head)
    V shape = (batch_size, n_keys, d_head)
    L shape = (barch, n_queries, n_keys)

    Q_tile @ K_tile.T
    (Q_TILE_SIZE, D) @ (D, K_TILE_SIZE)
    = (Q_TILE_SIZE, K_TILE_SIZE)


    Return the output 𝑶 and the logsumexp 𝐿.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=()
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_KEYS, D),
        strides=(stride_oq, stride_od),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=()
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, D),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, ),
        order=()
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero") # idx, (ROWS_TILE_SIZE, D_TILE_SIZE)
    output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    expl = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    for key_tile_index in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tile = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        v_tile = tl.load(V_block_ptr, boundary_check=(0,), padding_options="zero") 

        q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        score = tl.where(k_offsets[None, :] < N_KEYS, score, -1e6)
        score = tl.where(q_offsets[:, None] < N_QUERIES, score, -1e6)

        if IS_CAUSAL:
            score = tl.where(q_offsets[:, None] >= k_offsets[None, :], score, -1e6)

        m_prev = m
        m = tl.maximum(m, tl.max(score, axis=1))
        p_tile = tl.exp(score - m)
        alpha = tl.exp(m_prev, m)
        expl = alpha @ expl + tl.sum(p_tile, axis=1)
        output_tile = alpha[:, None] @ output_tile + tl.dot(p_tile, v_tile)

    factor = tl.exp(expl, -1)
    output = factor @ output_tile
    expsum = m + tl.log(expl)

    tl.store(O_block_ptr, output, boundary_check=(0, 1))
    tl.store(L_block_ptr, expsum, boundary_check=(0, 1))
    
def flash_backward_torch(q, k, v, o, L, do, is_causal):
    d = q.shape[-1]
    S = q @ k.transpose(-2, -1) / math.sqrt(d)

    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        S = torch.where(mask, S, -1e6)

    P = torch.exp(S - L.unsqueeze(-1))

    D = (o * do).sum(dim=-1)
    dV = P.transpose(-2, -1) @ do
    dP = do @ v.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = dS @ k / math.sqrt(d)
    dK = dS.transpose(-2, -1) @ q / math.sqrt(d)

    return dQ, dK, dV
    
class MyTritonFlashAttentionAutogradFunctionClass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal):

        ctx.save_for_backward(q, k, v)
        output, L = flash_fwd_kernel()
        ctx.save_for_backward(output, L)
        ctx.is_causal = is_causal

        return output, L


    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        compiled_flash_backward_torch = torch.compile(flash_backward_torch)

        dq, dk, dv = compiled_flash_backward_torch(q, k, v, o, L, grad_out, is_causal)

        return dq, dk, dv, None