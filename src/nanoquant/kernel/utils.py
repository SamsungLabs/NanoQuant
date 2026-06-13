# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
- weight packing
    - gemv
    - gemm
- kernel config auto-optimization
    - auto optimize for each matrix shape
    - set config for each shape
"""
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Lazy import gemlite to avoid hard dependency at import time
_gemlite = None


def _get_gemlite():
    """Lazy-load gemlite to avoid import-time errors with incompatible torch versions."""
    global _gemlite
    if _gemlite is None:
        import gemlite as _g
        _gemlite = _g
    return _gemlite


def get_kernel_function(namespace: str, dtype: torch.dtype, kernel_type: str):
    try:
        import binary_kernels
    except Exception as exc:
        raise ImportError("Could not import `binary_kernels`. Did you build the extension?\n"
                          f"Original error: {exc}") from exc

    mid = "_dyn" if kernel_type != "fixed" else ""
    op_name = f"{namespace}{mid}_forward"

    try:
        return getattr(torch.ops.binary_kernels, op_name)
    except AttributeError as exc:
        have = dir(torch.ops.binary_kernels) if hasattr(torch.ops, 'binary_kernels') else []
        raise AttributeError(f"Dispatcher op `{namespace}::{op_name}` not found. "
                             f"Available in torch.ops.binary_kernels: {have}") from exc


@torch.compile
def pack_binary_gemv(tensor: torch.Tensor, method: str = 'lsb_first') -> torch.Tensor:
    """
    Pack a {-1,+1} int8 matrix into row-major 32-bit words.

    This function supports multiple bit-ordering methods.

    Parameters
    ----------
    tensor : torch.Tensor[int8]  (n_rows, n_cols)
        Matrix to pack. Values must be exactly -1 or +1.
    method : str, optional
        Bit ordering within each 32-bit word.
        Supported methods: 'lsb_first', 'half2'. Default is 'lsb_first'.

    Returns
    -------
    torch.Tensor[int32]
        Packed tensor of shape ``(n_rows, words_per_row)``.

    Raises
    ------
    TypeError
        * Input tensor is not ``int8``.
    ValueError
        * ``method`` is not supported.
    """
    # Validate tensor dtype
    if tensor.dtype != torch.int8:
        raise TypeError("Input tensor must be int8.")

    # Get tensor dimensions
    n_rows, n_cols = tensor.shape
    device = tensor.device
    words_per_row: int = (n_cols + 31) // 32

    # Pad with +1 (which becomes bit-0 after the +1 → 0 / -1 → 1 conversion)
    padded_n_cols = words_per_row * 32
    if n_cols != padded_n_cols:
        pad_tensor = torch.ones((n_rows, padded_n_cols - n_cols), dtype=tensor.dtype, device=device)
        tensor_padded = torch.cat([tensor, pad_tensor], dim=1)
    else:
        tensor_padded = tensor

    # Convert to bits: +1 → 0 , -1 → 1
    bits = ((1 - tensor_padded) // 2).int().view(n_rows, words_per_row, 32)

    # Determine bit ordering based on method
    if method == 'lsb_first':
        powers = torch.arange(32, dtype=torch.int32, device=device)
    elif method == 'half2':
        # Interleaves bits from the first and second half of the 32-bit chunk
        # Order: [15, 31, 14, 30, ..., 0, 16]
        powers = torch.tensor([j for i in reversed(range(16)) for j in [i, i + 16]], dtype=torch.int32, device=device)
    else:
        raise ValueError(f"Unsupported method: '{method}'. Supported methods are 'lsb_first', 'half2'.")

    # Build per-bit weights
    wts = (2**powers).int()

    # Aggregate bits into 32-bit words
    packed = (bits * wts).sum(dim=2, dtype=torch.int32)
    return packed.contiguous()


@torch.compile
def unpack_binary_gemv(packed_tensor: torch.Tensor, original_shape: Tuple[int, int],
                       method: str = 'lsb_first') -> torch.Tensor:
    """
    Unpack a tensor from packed format back to {-1,+1} int8 matrix.

    This function unpacks a tensor that was previously packed using binary_packer,
    restoring it to its original {-1,+1} int8 format.

    Parameters
    ----------
    packed_tensor : torch.Tensor[int32]  (n_rows, words_per_row)
        Packed tensor to unpack. Must be a 2D tensor.
    original_shape : Tuple[int, int]
        The original (n_rows, n_cols) shape of the tensor before packing.
    method : str, optional
        Bit ordering method used for packing.
        Supported methods: 'lsb_first', 'half2'. Default is 'lsb_first'.

    Returns
    -------
    torch.Tensor[int8]
        Unpacked tensor of shape ``original_shape`` with values {-1,+1}.

    Raises
    ------
    ValueError
        * Packed tensor format is not supported (not 2D or incorrect shape).
        * ``method`` is not supported.
    """
    if packed_tensor.dim() != 2:
        raise ValueError(f"Unsupported packed tensor format. Expected 2D tensor, got {packed_tensor.dim()} dimensions.")

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 31) // 32

    if packed_tensor.shape != (n_rows, words_per_row):
        raise ValueError(
            f"Unsupported packed tensor shape. Expected ({n_rows}, {words_per_row}), got {packed_tensor.shape}.")

    device = packed_tensor.device

    # Determine bit ordering based on method
    if method == 'lsb_first':
        powers = torch.arange(32, dtype=torch.int32, device=device)
    elif method == 'half2':
        # This order matches the one used in the packer
        powers = torch.tensor([j for i in reversed(range(16)) for j in [i, i + 16]], dtype=torch.int32, device=device)
    else:
        raise ValueError(f"Unsupported method: '{method}'. Supported methods are 'lsb_first', 'half2'.")

    # Extract all 32 bits at once using vectorized operations.
    # The `powers` array determines which bit to right-shift for each position.
    # `(packed_tensor.unsqueeze(2) >> powers)` creates a (n_rows, words_per_row, 32) tensor
    # where the k-th element along the last dim is the packed word right-shifted by powers[k].
    # `& 1` extracts the least significant bit (the bit we want).
    bits = (packed_tensor.unsqueeze(2) >> powers) & 1

    # Reshape to a 2D padded tensor
    unpacked_padded = bits.view(n_rows, words_per_row * 32).to(torch.int8)

    # Trim to original size
    unpacked = unpacked_padded[:, :n_cols]

    # Convert from {0,1} to {-1,+1}
    # 0 -> 1 - 2*0 = +1
    # 1 -> 1 - 2*1 = -1
    return (1 - 2 * unpacked).to(torch.int8)


def pack_binary_marlin(
    weight: torch.Tensor,
    scale_in: torch.Tensor,
    scale_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    orig_shape = weight.shape
    reshape_args = (orig_shape[0] // 64,   4,  2,  8, orig_shape[1] // 16,  2,  4,  2) # yapf: disable
    [                          tile_out, o16, o8, o1,            tile_red, r8, r2, r1] = range(len(reshape_args)) # yapf: disable
    weight = weight.reshape(*reshape_args)
    weight = weight.permute(tile_red, tile_out, o1, r2, o16, r1, o8, r8).reshape(-1, 32).int()
    device = weight.device

    i8 = torch.arange(4, device=device).view(4, 1, 1)
    i4 = torch.arange(2, device=device).view(1, 2, 1)
    i1 = torch.arange(4, device=device).view(1, 1, 4)
    exponents = 15 - i1 + 16 * i4 - 4 * i8
    shift = (1 << exponents).to(torch.int32).flatten()

    weight = ((1 - weight) // 2 * shift).sum(-1).int()
    weight = weight.reshape(orig_shape[1] // 16, orig_shape[0] // 2).contiguous()
    scale_in = scale_in.reshape(scale_in.numel() // 16, 2, 4, 2)
    scale_in = scale_in.permute(0, 2, 1, 3).reshape(-1).contiguous()
    scale_out = scale_out.reshape(scale_out.numel() // 32, 4, 4, 2)
    scale_out = scale_out.permute(0, 2, 1, 3).reshape(-1).contiguous()

    # Calculate workspace size based on output scale
    workspace_size = scale_out.numel() // 128 * 16

    return weight, scale_in, scale_out, workspace_size


def pack_binary_marlin_fused(
    V_sign: torch.Tensor,
    U_sign: torch.Tensor,
    scale_g: torch.Tensor,
    scale_l: torch.Tensor,
    scale_h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack weights and scales for Marlin NanoQuant kernel.
    
    The Marlin NanoQuant kernel expects two weight matrices (U and V) and three scale tensors (g, l, h)
    packed in the Marlin format.
    
    Parameters
    ----------
    V_sign : torch.Tensor
        First weight matrix with shape [N, R] and values in {-1, +1}
    U_sign : torch.Tensor  
        Second weight matrix with shape [R, M] and values in {-1, +1}
    scale_g : torch.Tensor
        Input scale tensor with shape [N]
    scale_l : torch.Tensor
        Intermediate scale tensor with shape [R]
    scale_h : torch.Tensor
        Output scale tensor with shape [M]
    rank: int
        Rank of the low-rank binary matrices U, V
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        (U_bits, V_bits, scale_g_marlin, scale_l_marlin, scale_h_marlin, workspace_size)
    """
    # get device
    dev = V_sign.device
    # get shapes of low-rank binary matrices
    # get matrix dims
    N = scale_g.shape[0]
    R = scale_l.shape[0] if scale_l is not None else U_sign.shape[0]
    M = scale_h.shape[0]
    # make dimensions multiple of 32
    p = 32
    if N % p != 0: N = ((N // p) + 1) * p
    if R % p != 0: R = ((R // p) + 1) * p
    if M % p != 0: M = ((M // p) + 1) * p
    # make rank multiple of 256
    rank_p = 256
    rank_marlin = (R + rank_p - 1) // rank_p * rank_p

    if rank_marlin > R:
        pad_amount = rank_marlin - R
        V_sign = torch.cat([V_sign, torch.ones(N, pad_amount, device=dev, dtype=torch.int8)], dim=1)
        U_sign = torch.cat([U_sign, torch.ones(pad_amount, M, device=dev, dtype=torch.int8)], dim=0)
        if scale_l is not None:
            scale_l = torch.cat([scale_l, torch.zeros(pad_amount, device=dev, dtype=scale_l.dtype)], dim=0)

    # Pack U_sign and V_sign using the marlin_onebit_packer
    V_bits, scale_g_marlin, scale_l_marlin, workspace_size_v = marlin_onebit_packer(V_sign.T, scale_g, scale_l)
    # scale_l is used as the output scale of the first stage
    # here it's passed as the placeholder for input scale parameter, so the returned scale_in is ignored
    U_bits, _, scale_h_marlin, workspace_size_u = marlin_onebit_packer(U_sign.T, scale_l, scale_h)

    # Use the maximum workspace size from both stages
    workspace_size = max(workspace_size_v, workspace_size_u)

    workspace = torch.zeros(workspace_size, device=dev, dtype=torch.int32)

    return V_bits.contiguous(), U_bits.contiguous(), scale_g_marlin, scale_l_marlin, scale_h_marlin, workspace


def gemlite_packer(
    *,
    U: torch.Tensor,  # [N, rank]
    V: torch.Tensor,  # [rank, K]
    scale_pre: torch.Tensor,  # [K] or [1,K]
    scale_post: torch.Tensor,  # [N] or [1,N]
    scale_mid: Optional[torch.Tensor] = None,  # [rank] or [1,rank] or None
    pad_multiple: Union[int, Tuple[int, int]] = 128,
    dtype: torch.dtype = torch.float16,
    config_path: Optional[str] = None,
    save_config: bool = False,
) -> Dict[str, Any]:
    """
    Pad + pack NanoQuant's two-stage binary weights into GemLite layers.

    - Pads V and U to multiples of pad_multiple (default 128) using +1 padding.
    - Pads scale_mid to rankp with 0 in padded tail (kills padded ranks).
    - Leaves scale_pre/scale_post unpadded (apply them in original space; pad x with zeros, slice y back).

    Returns:
      {
        "V_linear": GemLiteLinear (packed, padded)  maps [M,Kp]->[M,rankp]
        "U_linear": GemLiteLinear (packed, padded)  maps [M,rankp]->[M,Np]
        "scale_pre0": 1D [K0] (unpadded)
        "scale_mid_pad": 1D [rankp] (padded tail=0)
        "scale_post0": 1D [N0] (unpadded)
        "pad": {K0,Kp,rank0,rankp,N0,Np,gs1,gs2}
      }
    """

    # -----------------------
    # helpers
    # -----------------------
    def _to_1d(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 2 and x.shape[0] == 1:
            return x.squeeze(0)
        if x.ndim == 1:
            return x
        raise ValueError(f"{name} must be 1D or [1,D], got {tuple(x.shape)}")

    def _ceil_to(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    def _pad_2d_uint01(W_uint8: torch.Tensor, out_multiple: int, in_multiple: int, pad_val_uint01: int):
        out0, in0 = W_uint8.shape
        outp = _ceil_to(out0, out_multiple)
        inp = _ceil_to(in0, in_multiple)
        if outp == out0 and inp == in0:
            return W_uint8.contiguous(), outp, inp
        Wp = F.pad(W_uint8, (0, inp - in0, 0, outp - out0), value=int(pad_val_uint01))
        return Wp.contiguous(), outp, inp

    def _pad_1d(x: torch.Tensor, new_len: int, pad_val: float) -> torch.Tensor:
        if x.numel() == new_len:
            return x
        if x.numel() > new_len:
            raise ValueError(f"Cannot shrink in _pad_1d: {x.numel()} -> {new_len}")
        return F.pad(x, (0, new_len - x.numel()), value=float(pad_val))

    def _pick_group_size_fixed(in_features: int) -> int:
        # Same policy as your file: prefer 128 then 64 then 32, requiring n_groups >= 2.
        for gs in (128, 64, 32):
            if in_features >= 2 * gs:
                return gs
        raise ValueError(f"in_features={in_features} too small for GemLite grouped mode (need >= 64).")

    # -----------------------
    # checks
    # -----------------------
    if V.ndim != 2 or U.ndim != 2:
        raise ValueError(f"Expected 2D V/U. Got V={tuple(V.shape)} U={tuple(U.shape)}")

    rank0, K0 = V.shape
    N0, rank0b = U.shape
    if rank0b != rank0:
        raise ValueError(f"U second dim must match V first dim. Got U={tuple(U.shape)} V={tuple(V.shape)}")

    device = V.device
    if device.type != "cuda":
        raise RuntimeError("GemLite packing requires CUDA tensors for V/U.")
    if U.device != device:
        raise ValueError("U and V must be on the same device.")

    scale_pre0 = _to_1d(scale_pre, "scale_pre").to(device=device, dtype=dtype).contiguous()
    scale_post0 = _to_1d(scale_post, "scale_post").to(device=device, dtype=dtype).contiguous()
    if scale_pre0.numel() != K0:
        raise ValueError(f"scale_pre length must be K0={K0}, got {scale_pre0.numel()}")
    if scale_post0.numel() != N0:
        raise ValueError(f"scale_post length must be N0={N0}, got {scale_post0.numel()}")

    if scale_mid is None:
        scale_mid0 = torch.ones(rank0, device=device, dtype=dtype)
    else:
        scale_mid0 = _to_1d(scale_mid, "scale_mid").to(device=device, dtype=dtype).contiguous()
        if scale_mid0.numel() != rank0:
            raise ValueError(f"scale_mid length must be rank0={rank0}, got {scale_mid0.numel()}")

    # -----------------------
    # config load
    # -----------------------
    if config_path is None:
        config_path = os.environ.get("GEMLITE_CONFIG_PATH", None)
    if config_path and os.path.exists(config_path):
        try:
            _get_gemlite().load_config(config_path)
        except Exception:
            pass

    # -----------------------
    # pad multiples
    # -----------------------
    if isinstance(pad_multiple, (tuple, list)):
        pad_out_mult = int(pad_multiple[0])
        pad_in_mult = int(pad_multiple[1])
    else:
        pad_out_mult = int(pad_multiple)
        pad_in_mult = int(pad_multiple)

    # -----------------------
    # convert {-1,+1} -> uint8 {0,1} and pad
    # -----------------------
    V_pm1 = V.data.sign()
    V_pm1[V_pm1 == 0] = 1
    U_pm1 = U.data.sign()
    U_pm1[U_pm1 == 0] = 1

    V_uint8 = (V_pm1 > 0).to(torch.uint8)
    U_uint8 = (U_pm1 > 0).to(torch.uint8)

    # V: [rankp, Kp]
    V_pad, rankp, Kp = _pad_2d_uint01(V_uint8, pad_out_mult, pad_in_mult, pad_val_uint01=1)
    # U: [Np, rankp?] (pad to multiples; then force columns to rankp)
    U_pad, Np, rankp2 = _pad_2d_uint01(U_uint8, pad_out_mult, pad_out_mult, pad_val_uint01=1)
    if rankp2 != rankp:
        if rankp2 < rankp:
            U_pad = F.pad(U_pad, (0, rankp - rankp2, 0, 0), value=1).contiguous()
            rankp2 = rankp
        else:
            raise RuntimeError(f"Unexpected: U padded rank {rankp2} > V padded rank {rankp}")
    if rankp2 != rankp:
        raise RuntimeError(f"Internal pad mismatch: rankp={rankp} rankp2={rankp2}")

    scale_mid_pad = _pad_1d(scale_mid0, rankp, pad_val=0.0).contiguous()

    # -----------------------
    # group sizes / constraints
    # -----------------------
    gs1 = _pick_group_size_fixed(K0)
    gs2 = _pick_group_size_fixed(rank0)
    if (Kp % gs1) != 0 or (Kp // gs1) < 2:
        raise RuntimeError(f"Stage1 invalid groups: Kp={Kp}, gs1={gs1}")
    if (rankp % gs2) != 0 or (rankp // gs2) < 2:
        raise RuntimeError(f"Stage2 invalid groups: rankp={rankp}, gs2={gs2}")

    gemlite_input_dtype = {
        torch.float16: _get_gemlite().DType.FP16,
        torch.bfloat16: _get_gemlite().DType.BF16,
    }[dtype]

    # -----------------------
    # build + pack layers
    # -----------------------
    def _build_gemlite_layer(W_uint8_: torch.Tensor, in_features: int, out_features: int, group_size: int):
        n_groups = in_features // group_size
        layer = _get_gemlite().GemLiteLinear(
            W_nbits=1,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            input_dtype=gemlite_input_dtype,
            output_dtype=gemlite_input_dtype,
        ).to(device)

        # Exact {-1,+1} dequant: (W_q - 0.5) * 2.0
        scales_g = torch.full((out_features, n_groups), 2.0, dtype=dtype, device=device)
        zeros_g = torch.full((out_features, n_groups), 0.5, dtype=dtype, device=device)
        layer.pack(W_uint8_.contiguous(), scales=scales_g, zeros=zeros_g, bias=None, fma_mode=False)

        if getattr(layer, "channel_scale_mode", None) != 0:
            raise RuntimeError(f"GemLite channel_scale_mode expected 0, got {layer.channel_scale_mode}")
        if getattr(layer, "W_group_mode", None) != 3:
            raise RuntimeError(f"GemLite W_group_mode expected 3, got {layer.W_group_mode}")

        return layer

    V_linear = _build_gemlite_layer(V_pad, in_features=Kp, out_features=rankp, group_size=gs1)
    U_linear = _build_gemlite_layer(U_pad, in_features=rankp, out_features=Np, group_size=gs2)

    if save_config and config_path:
        try:
            _get_gemlite().cache_config(config_path)
        except Exception:
            pass

    return {
        "V_linear": V_linear,
        "U_linear": U_linear,
        "scale_pre0": scale_pre0,  # [K0]
        "scale_mid_pad": scale_mid_pad,  # [rankp]
        "scale_post0": scale_post0,  # [N0]
        "pad": {
            "K0": K0,
            "Kp": Kp,
            "rank0": rank0,
            "rankp": rankp,
            "N0": N0,
            "Np": Np,
            "gs1": gs1,
            "gs2": gs2,
        },
    }


# =============================================================================
# Aliases for Backward Compatibility
# =============================================================================
binary_packer = pack_binary_gemv
binary_unpacker = unpack_binary_gemv
marlin_onebit_packer = pack_binary_marlin
marlin_nanoquant_packer = pack_binary_marlin_fused
gemlite_nanoquant_packer = gemlite_packer
