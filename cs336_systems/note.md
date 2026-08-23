Common Attention Optimization Algorithm
1. FlashAttention Tiling
2. PagedAttention KV cache and virtual memory
    borrows the virtual memory paging concept from operating systems. It divides the KV cache into small, fixed-size blocks (pages) that can live in non-contiguous physical GPU memory and uses a block table to map them