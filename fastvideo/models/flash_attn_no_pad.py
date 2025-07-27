from einops import rearrange
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    # Flash attention not available on macOS
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_qkvpacked_func = None
    pad_input = None
    unpad_input = None


def flash_attn_no_pad(qkv,
                      key_padding_mask,
                      causal=False,
                      dropout_p=0.0,
                      softmax_scale=None):
    if not FLASH_ATTN_AVAILABLE:
        # Fallback to standard attention for macOS
        import torch
        import torch.nn.functional as F
        
        batch_size, seqlen, three, nheads, head_dim = qkv.shape
        
        # Split qkv
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [b, s, h, d]
        
        # Compute attention
        scale = softmax_scale if softmax_scale is not None else (head_dim ** -0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [b, s, h, s]
        
        if causal:
            # Apply causal mask
            mask = torch.triu(torch.ones(seqlen, seqlen, device=qkv.device), diagonal=1)
            attn_weights = attn_weights.masked_fill(mask.bool().unsqueeze(0).unsqueeze(2), float('-inf'))
            
        # Apply padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [b, s] - True for valid positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [b, 1, 1, s]
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
            
        output = torch.matmul(attn_weights, v)  # [b, s, h, d]
        return output
    
    # Original flash attention implementation
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask)

    x_unpad = rearrange(x_unpad,
                        "nnz (three h d) -> nnz three h d",
                        three=3,
                        h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices,
                  batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output
