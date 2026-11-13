import math

import torch

F64 = torch.float64


def plane_angles(n_planes, device=None):
    return torch.arange(n_planes, dtype=F64, device=device) * (math.pi / n_planes) - 0.5 * math.pi


def plane_traction(eps, lam, mu, n_planes):
    b = plane_angles(n_planes, eps.device)
    tr = eps[:, 0] + eps[:, 1]
    sxx = lam * tr + 2 * mu * eps[:, 0]
    syy = lam * tr + 2 * mu * eps[:, 1]
    sxy = mu * eps[:, 2]
    s2, c2 = torch.sin(2 * b), torch.cos(2 * b)
    sm, sd = 0.5 * (sxx + syy)[:, None], 0.5 * (sxx - syy)[:, None]
    sn = sm - sd * c2 - sxy[:, None] * s2
    tau = -sd * s2 + sxy[:, None] * c2
    return sn, tau


def accumulate(eps_cycle, lam, mu, n_planes):
    sn0, tau0 = plane_traction(eps_cycle[-1], lam, mu, n_planes)
    O = torch.zeros_like(sn0)
    E = torch.zeros_like(sn0)
    for eps in eps_cycle:
        sn1, tau1 = plane_traction(eps, lam, mu, n_planes)
        gate = 0.5 * (sn0.clamp(min=0) + sn1.clamp(min=0))
        dt = tau1 - tau0
        O += gate * dt
        E += gate * dt.abs()
        sn0, tau0 = sn1, tau1
    return O, E


def odd_tensor(O, E):
    n = O.shape[1]
    b = plane_angles(n, O.device)
    w = O / n
    t11 = -0.5 * (w * torch.sin(2 * b)).sum(1)
    t12 = 0.5 * (w * torch.cos(2 * b)).sum(1)
    T = torch.stack([torch.stack([t11, t12], 1), torch.stack([t12, -t11], 1)], 1)
    return T, E.max(1).values


def structural_tensor(T, E, kappa, lmax, intact=None, norm="local"):
    sel = E if intact is None or not bool(intact.any()) else E[intact]
    Emax = max(float(sel.max()), 1e-30)
    if norm == "local":
        Tref = torch.clamp(E, min=1e-3 * Emax)
    else:
        Tref = torch.full_like(E, Emax)
    S = -kappa * T / Tref[:, None, None]
    lam = torch.sqrt(S[:, 0, 0] ** 2 + S[:, 0, 1] ** 2)
    cap = torch.clamp(lmax / torch.clamp(lam, min=1e-30), max=1.0)
    S, lam = S * cap[:, None, None], lam * cap
    sh = torch.where(lam > 1e-12, torch.sinh(lam) / torch.clamp(lam, min=1e-12), torch.ones_like(lam))
    I2 = torch.eye(2, dtype=F64, device=lam.device)
    A = torch.cosh(lam)[:, None, None] * I2 + sh[:, None, None] * S
    return A, S


def damage_stiffness(G, A, Gc, ell, H, Me, vol):
    GG = torch.einsum("eia,eab,ejb->eij", G, A, G)
    return ((Gc * ell)[:, None, None] * GG + (Gc / ell + 2 * H)[:, None, None] * Me) * vol[:, None, None]
