# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
ADMM from DBF. We remove support for sparsity, but the ADMM logic is kept.
https://github.com/usamec/double_binary/blob/master/raw_stuff/compress/Compress-Llama-2-7B-tar15.ipynb
"""

import torch


@torch.no_grad()
def power_iteration(A, num_iters=5):
    # Start with a random vector on the appropriate device
    n = A.shape[1]
    v = torch.randn(n, dtype=A.dtype, device=A.device)
    v = v / torch.norm(v)

    for _ in range(num_iters):
        # Multiply A*v
        u = torch.mv(A, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm

        # Multiply A^T*u
        v = torch.mv(A.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            break
        v = v / v_norm

    # Estimate the dominant singular value as ||A*v||
    u_unnorm = torch.mv(A, v)
    sigma = torch.norm(u_unnorm)
    # The left singular vector corresponding to sigma:
    u = u_unnorm / sigma
    return u, sigma, v


@torch.no_grad()
def svd_abs(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    apx = s * torch.outer(u, v)
    return apx * Sg


@torch.no_grad()
def svd_abs2(W):
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    return u * s, Sg, v


def _admm_solve_step(X, Y, Z, U, rho_start, reg=3e-2, inner_iters=3):
    """
    ADMM solver that mimics the original `find_other2` logic.
    It uses `rho_start` for the first step and a fixed `rho=1` for subsequent steps.
    """
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))

    XX = X.T.matmul(X)
    XX += torch.diag(torch.ones_like(XX.diag())) * XX.diag().mean() * reg
    XY = X.T.matmul(Y)

    # Pre-calculate inverse matrices for both rho values
    XXinv_start = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device) * rho_start)
    XXinv_fixed = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device) * 1.0)

    # First step uses rho_start
    Factor = XXinv_start.matmul(XY + rho_start * (Z - U))

    # Subsequent steps use fixed rho=1
    for _ in range(inner_iters - 1):
        Z = svd_abs(Factor + U)
        U = U + (Factor - Z)
        Factor = XXinv_fixed.matmul(XY + 1.0 * (Z - U))

    Z = svd_abs(Factor + U)
    U = U + (Factor - Z)

    Z, U, Factor = (t.to(orig_dtype) for t in (Z, U, Factor))
    return Z, U, Factor


def factorize_admm_dbf(W, i_norm, o_norm, mid_rank, iters=260, is_transpose=False, eps=1e-8, use_latent=False):
    """
    Decomposes the weight matrix W into two binary matrices A and B using ADMM.
    Assumes W has the shape (out_features, in_features).
    """
    if is_transpose:
        # For layers like fc2/down_proj where in > out, process the transpose
        results = factorize_admm_dbf(
            W.T,
            o_norm,
            i_norm,
            mid_rank,
            iters,
            is_transpose=False,
            eps=eps,
            use_latent=use_latent,
        )
        # Return A, B in the correct order and restore the original weight shape
        return {
            "W_final": results["W_final"].T,
            "A": results["B"],
            "B": results["A"],
            "A_latent": results["B_latent"],
            "B_latent": results["A_latent"],
            # Swap pre and post for the transpose
            "scale_pre": results["scale_post"],
            "scale_mid": results["scale_mid"],
            "scale_post": results["scale_pre"],
        }

    device = W.device
    out_features, in_features = W.shape

    # Re-scale norms by a heuristic factor (128) to compensate for the division by
    # n_samples in calibration. This restores the magnitude range the ADMM solver
    # expects, preventing numerical instability/underflow.
    norm_i = (i_norm).sqrt().clamp(eps)
    norm_o = (o_norm).sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i * norm_o

    Az = torch.randn((out_features, mid_rank), device=device)
    Au = torch.zeros_like(Az)
    Bz = torch.randn((mid_rank, in_features), device=device)
    Bu = torch.zeros_like(Bz)

    for itt in range(iters):
        # Calculate rho_start, which changes over iterations
        rho_start = min(1.0, itt / (iters - 3))**3 if iters > 3 else 1.0

        # Update A (W.T = B.T @ A.T)
        # The asymmetric scaling is kept as it is part of the original's design
        mid_norm_b = Bz.norm(dim=1).clamp(eps)
        X_A = Bz.T / mid_norm_b
        Az_T, Au_T, Als_T = _admm_solve_step(X_A, W_norm.T, Az.T, Au.T, rho_start)
        Az, Au, Als = Az_T.T, Au_T.T, Als_T.T

        # Update B (W = A @ B)
        mid_norm_a = Az.norm(dim=0).clamp(eps)
        X_B = Az / mid_norm_a
        Bz, Bu, Bls = _admm_solve_step(X_B, W_norm, Bz, Bu, rho_start)

    # --- 1. Final Scaling and Normalization ---
    A_final_unbalanced = Az / norm_o
    B_final_unbalanced = Bz / norm_i

    A_latent_unbalanced = (Als + Au) / norm_o
    B_latent_unbalanced = (Bls + Bu) / norm_i

    # --- 2. Final Norm Balancing ---
    # Balance the norms of A and B for numerical stability
    norm_A = A_final_unbalanced.norm().clamp(eps)
    norm_B = B_final_unbalanced.norm().clamp(eps)
    balance_factor = (norm_B / norm_A).sqrt()

    A_final = A_final_unbalanced * balance_factor
    B_final = B_final_unbalanced / balance_factor

    A_latent = A_latent_unbalanced * balance_factor
    B_latent = B_latent_unbalanced / balance_factor

    # The mid_scale compensates for the scaling applied to B's input (Az)
    final_mid_scale_factor = Az.norm(dim=0).clamp(eps)
    mid_scale = 1 / final_mid_scale_factor

    W_final = (A_final * mid_scale).matmul(B_final)

    # --- Extracting the 3 scales for NanoQuantLinear ---
    # Calculate scales based on mean magnitudes
    A = A_final.T
    B = B_final

    u1, b1, v1 = svd_abs2(B.float())
    u2, b2, v2 = svd_abs2(A.float())

    scale_pre = v1
    scale_mid = u1 * mid_scale * u2
    scale_post = v2

    return {
        "W_final": W_final,
        "A": b2,  # (mid, out)
        "B": b1,  # (mid, in)
        "A_latent": b2 if not use_latent else A_latent.T,
        "B_latent": b1 if not use_latent else B_latent,
        "scale_pre": scale_pre,
        "scale_mid": scale_mid,
        "scale_post": scale_post,
    }
