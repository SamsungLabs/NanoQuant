# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Local registry for rho schedulers
RHO_SCHEDULER_REGISTRY = {}


@torch.no_grad()
def power_iteration(A, num_iters=5):
    """
    Power iteration for top singular triplet (u, sigma, v) of A.
    """
    n = A.shape[1]
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    v = v / torch.norm(v)

    At = A.mT  # view; reuse
    for _ in range(num_iters):
        u = torch.mv(A, v)
        u = u / u.norm()

        v = torch.mv(At, u)
        v = v / v.norm()

    u_unnorm = torch.mv(A, v)
    sigma = torch.norm(u_unnorm)
    u = u_unnorm / sigma
    return u, sigma, v


@torch.no_grad()
def svid(W, inner_iters=5, eps=1e-12):
    """
    Sign-Value-Independent Decomposition (SVID).
    Returns u, v, Sg where Sg is sign matrix of W.
    """
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), inner_iters)
    u = u * s
    return u, v, Sg


@torch.no_grad()
def rank1_approx(W, inner_iters=5, eps=1e-12):
    """
    Rank-1 approximation using SVID results.
    """
    u, v, Sg = svid(W, inner_iters, eps)
    apx = torch.outer(u, v)
    return apx * Sg


@torch.no_grad()
def _admm_solve_step(X, Y, Z, U, rho, reg, eps=1e-12):
    """
    Solves one step of ADMM robustly using a stabilized Cholesky decomposition.
    Solve: (X^T X + stabilizer*I) * Factor = X^T Y + rho*(Z-U)
    """
    orig_dtype = X.dtype
    X, Y, Z, U = (t.to(torch.float32) for t in (X, Y, Z, U))

    Xt = X.mT  # view, no materialization
    system_matrix = Xt @ X  # (k,k)
    system_matrix = 0.5 * (system_matrix + system_matrix.mT)

    # stabilizer on diagonal
    diag_mean = system_matrix.diagonal().mean().abs()
    stabilizer = torch.clamp(rho * diag_mean + reg, min=eps)
    system_matrix.diagonal().add_(stabilizer)

    rhs = (Xt @ Y) + rho * (Z - U)

    # Fast path: cholesky_ex gives info instead of exception
    L, info = torch.linalg.cholesky_ex(system_matrix, upper=False)

    if info.item() == 0:
        Factor = torch.cholesky_solve(rhs, L, upper=False)
    else:
        # Rare fallback
        Factor = torch.linalg.solve(system_matrix, rhs)

    return Factor.to(orig_dtype)


@torch.no_grad()
def factorize_admm_nanoquant(
    W,
    i_norm,
    o_norm,
    mid_rank,
    outer_iters=400,
    inner_iters=5,
    reg=3e-2,
    is_transpose=False,
    eps=1e-12,
    rho_scheduler='cubic',
    print_admm_steps=False,
):
    """
    Decomposes the weight matrix W into two binary matrices A and B using ADMM.
    Assumes W has the shape (out_features, in_features).
    Post-processing extracts Scales and Binary-compatible float matrices based on
    Mean Magnitude extraction (Scale-Binary-Binary-Scale).

    Args:
        W: Weight matrix to decompose
        i_norm: Input norm
        o_norm: Output norm  
        mid_rank: Middle rank for factorization
        outer_iters: Number of outer iterations
        inner_iters: Number of inner iterations
        reg: Regularization parameter
        is_transpose: Whether to transpose the weight matrix
        eps: Small epsilon value to prevent division by zero and numerical instability
        rho_scheduler: Rho scheduler name. Available: 
                       ['cubic', 'linear', 'logistic', 'exp_decay', 'exp_growth']
        print_admm_steps: Whether to print intermediate ADMM steps
    """
    if is_transpose:
        results = factorize_admm_nanoquant(W.mT, o_norm, i_norm, mid_rank, outer_iters, inner_iters, reg, False, eps,
                                           rho_scheduler, print_admm_steps)
        return {
            "W_final": results["W_final"].mT,
            "A": results["B"],
            "B": results["A"],
            "A_latent": results["B_latent"],
            "B_latent": results["A_latent"],
            "scale_pre": results["scale_post"],
            "scale_post": results["scale_pre"],
        }

    device = W.device
    out_features, in_features = W.shape

    norm_i = i_norm.sqrt().clamp(eps)
    norm_o = o_norm.sqrt().clamp(eps).unsqueeze(1)
    W_norm = W * norm_i.unsqueeze(0) * norm_o

    # we remove SVD-based init, since random init is (1) faster (2) shows on-par or better performance
    A_ls = torch.randn((out_features, mid_rank), device=device, dtype=W.dtype)
    B_ls = torch.randn((mid_rank, in_features), device=device, dtype=W.dtype)

    A_z, B_z = A_ls, B_ls
    if outer_iters > 0:
        A_z = rank1_approx(A_ls, inner_iters, eps)
        B_z = rank1_approx(B_ls, inner_iters, eps)

    if print_admm_steps:
        A_z_old = A_z.clone()
        B_z_old = B_z.clone()

    A_u = A_ls - A_z
    B_u = B_ls - B_z

    rho_scheduler_func = RHO_SCHEDULER_REGISTRY[rho_scheduler]

    for itt in range(outer_iters):
        rho = rho_scheduler_func(itt / outer_iters)

        # 1) X-update
        mid_norm_b = B_z.norm(dim=1).clamp(eps)
        X_A = B_z.mT / mid_norm_b  # (in, mid)
        # W_norm.T uses view; keep it
        A_ls = _admm_solve_step(X_A, W_norm.mT, A_z.mT, A_u.mT, rho, reg, eps).mT

        mid_norm_a = A_z.norm(dim=0).clamp(eps)
        X_B = A_z / mid_norm_a  # (out, mid)
        B_ls = _admm_solve_step(X_B, W_norm, B_z, B_u, rho, reg, eps)

        # 2) Z-update
        target_A = A_ls + A_u
        target_B = B_ls + B_u
        A_z = rank1_approx(target_A, inner_iters, eps)
        B_z = rank1_approx(target_B, inner_iters, eps)

        # 3) U-update
        A_u.add_(A_ls - A_z)
        B_u.add_(B_ls - B_z)

        if print_admm_steps:
            if (itt == 0 or (itt + 1) % 100 == 0 or itt == outer_iters - 1):
                r_A = torch.norm(A_ls - A_z).item()
                r_B = torch.norm(B_ls - B_z).item()
                primal_res = r_A + r_B

                s_A = torch.norm(rho * (A_z - A_z_old)).item()
                s_B = torch.norm(rho * (B_z - B_z_old)).item()
                dual_res = s_A + s_B

                mid = B_z.norm(dim=1).clamp(eps)
                # (A_z / mid) @ B_z  ->  F.linear(A_z / mid, B_z.T)
                pred = F.linear(A_z / mid, B_z.mT)
                curr_loss = (W_norm - pred).norm().item()
                normalized_err = (curr_loss**2) / (W_norm.norm()**2).clamp(eps)

                print(f"\t\t[ADMM Step {itt+1:04d}/{outer_iters:04d}] Loss: {normalized_err:.5e} | "
                      f"Primal(r): {primal_res:.5e} | Dual(s): {dual_res:.5e} | Rho: {rho:.4f}")

            A_z_old.copy_(A_z)
            B_z_old.copy_(B_z)

    # Final export
    A_final_unbalanced = A_z / norm_o
    B_final_unbalanced = B_z / norm_i

    A_latent = (A_ls + A_u) / norm_o
    B_latent = (B_ls + B_u) / norm_i

    A_unbalanced = A_z / norm_o
    B_unbalanced = B_z / norm_i

    A_latent_unb = (A_ls + A_u) / norm_o
    B_latent_unb = (B_ls + B_u) / norm_i

    norm_A = A_unbalanced.norm().clamp(eps)
    norm_B = B_unbalanced.norm().clamp(eps)
    balance_factor = (norm_B / norm_A).sqrt()

    A_final = A_unbalanced * balance_factor
    B_final = B_unbalanced / balance_factor
    A_latent = A_latent_unb * balance_factor
    B_latent = B_latent_unb / balance_factor

    scale_factor = 1.0
    if outer_iters > 0:
        scale_factor = 1.0 / A_z.norm(dim=0).clamp(eps)
    A_final = A_final * scale_factor

    scale_pre = B_final.abs().mean(dim=0).view(1, -1)
    scale_post = A_final.abs().mean(dim=1).view(1, -1)

    # W_final = A_final @ B_final  -> F.linear(A_final, B_final.T)
    W_final = F.linear(A_final, B_final.mT)

    return {
        "W_final": W_final,
        "A": A_final.mT,  # (mid, out)
        "B": B_final,  # (mid, in)
        "A_latent": A_latent.mT,  # (mid, out)
        "B_latent": B_latent,  # (mid, in)
        "scale_pre": scale_pre,
        "scale_post": scale_post,
    }


def cubic(x):
    """Cubic rho scheduler with early iterations protection."""
    return min(1.0, x)**3


def linear(x):
    """Linear rho scheduler."""
    return x


def logistic(x, k=5):
    """Logistic rho scheduler."""
    return 1 / (1 + np.exp(-k * (x - 0.5)))


def exp_decay(x, k=5):
    """Exponential decay rho scheduler."""
    return (1 - np.exp(-k * x)) / (1 - np.exp(-k))


def exp_growth(x, k=5):
    """Exponential growth rho scheduler."""
    return (np.exp(k * x) - 1) / (np.exp(k) - 1)


# Register the scheduler functions
RHO_SCHEDULER_REGISTRY.update({
    'cubic': cubic,
    'linear': linear,
    'logistic': logistic,
    'exp_decay': exp_decay,
    'exp_growth': exp_growth,
})
