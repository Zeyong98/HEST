"""Geodesic steering inside the hyperbolic probe ("chart").

At a token with hidden state h (layer L), the probe E maps it into the Poincare ball: z = E(h).
The step is taken IN THE BALL, along the geodesic, with Mobius addition:

    u  = -grad^R s(z)                     unit-speed geodesic direction (s = Busemann b_xi or radius r)
    z' = exp_z(t u) = z (+)_c  tanh(sqrt(c) t / 2) / sqrt(c) * u_hat        <- Mobius addition
    find dh  with  E(h + dh) ~= z'        (iterated least-squares pullback through J = dE/dh)
    h <- h + dh,  clipped to |dh| <= rho_max |h|

`t` is the step length in HYPERBOLIC DISTANCE, i.e. in the probe's own units; a negative t steps away from the
ideal point (toward the hesitant side).

A probe trained with `scripts/train_probe.py --geometry euclid` (kappa = 0, the geometry ablation of the paper) takes
the c -> 0 limit of the same step: exp0 is the identity, z' = z + (t/2) u_hat, and the Jacobian loses the factor of exp0.
"""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hest.probe import ChartV3, radius, mobius_add, dist  # noqa: E402

EPS = 1e-6


class ChartField:
    def __init__(self, path, target="ideal", t=1.0, rho_max=1.0, iters=4, eps=1e-3):
        pk = torch.load(path, map_location="cpu", weights_only=False)
        self.geometry = pk.get("geometry", "hyp")                 # "hyp" (HEST, all released probes) | "euclid" (kappa = 0)
        self.ch = ChartV3(self.geometry, pk["score"], hid=pk["hid"], d_in=pk["d_in"], d_out=pk.get("d_out", 128)).cuda()
        self.ch.load_state_dict(pk["state_dict"], strict=False)
        self.ch.eval()
        for p in self.ch.parameters():
            p.requires_grad_(False)
        self.mu, self.sd = pk["mu"].float().cuda()[0], pk["sd"].float().cuda()[0]
        self.kap = self.ch.kappa()
        self.c = float(-self.kap)
        self.target, self.t, self.rho_max, self.iters, self.eps = target, float(t), float(rho_max), int(iters), eps
        self.name = os.path.basename(path)
        self.layer = pk.get("layer")
        if self.layer is None:                                   # older charts: recover the layer from the filename
            m = re.search(r"_L(\d+)", self.name)
            self.layer = int(m.group(1)) if m else None
        self.log = []                      # (reading before, reading after, hyperbolic distance moved, |dh|/|h|)

    # ---------------- chart ----------------
    def _z(self, h):
        return self.ch.embed(self.ch.encode(((h - self.mu) / self.sd)[None, :]))[0]

    def _reading(self, z):
        if self.target == "ideal":
            return self.ch.busemann(z[None, :])[0]
        return radius(z[None, :], self.kap)[0]

    def _geodesic_target(self, z):
        """z' = exp_z(t * u), u = -grad^R reading (unit speed), via Mobius addition."""
        zz = z.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(self._reading(zz), zz)[0]          # Euclidean gradient in ball coordinates
        if self.c == 0.0:
            # kappa = 0: the c -> 0 limit of the Mobius step below is z' = z + (t/2) u_hat (dist = 2|x - y| in this
            # parameterisation); for the radius readout the step never passes the origin.
            u = -g
            step = self.t / 2.0
            if self.target != "ideal" and step > 0:
                step = min(step, float(z.detach().norm()))
            return z.detach() + step * (u / u.norm().clamp_min(EPS))
        lam2 = ((1.0 - self.c * (z * z).sum()) ** 2) / 4.0          # (1/lambda_z^2): Riemannian = lam2 * Euclidean
        u = -(lam2 * g)
        un = u.norm().clamp_min(EPS)
        sc = self.c ** 0.5
        v = (torch.tanh(torch.tensor(sc * self.t / 2.0, device=z.device)) / sc) * (u / un)
        return mobius_add(z.detach()[None, :], v[None, :], self.kap)[0]

    # ---------------- analytic Jacobian ----------------
    def _jac(self, h):
        """J = dE/dh in closed form (encoder = standardise -> Linear -> GELU -> Linear -> exp0), ~200x faster than jacrev."""
        W1, b1 = self.ch.mlp[0].weight, self.ch.mlp[0].bias
        W2, b2 = self.ch.mlp[2].weight, self.ch.mlp[2].bias
        x = (h - self.mu) / self.sd
        a1 = W1 @ x + b1
        # GELU (exact, erf form) derivative
        cdf = 0.5 * (1.0 + torch.erf(a1 / 2 ** 0.5))
        pdf = torch.exp(-0.5 * a1 * a1) / (2 * torch.pi) ** 0.5
        dg = cdf + a1 * pdf
        v = W2 @ (cdf * a1) + b2
        scale = self.ch.scale.clamp_min(EPS)
        if self.c == 0.0:                                           # kappa = 0: exp0 is the identity, dz/du = I
            return ((W2 * dg[None, :]) @ (W1 / self.sd[None, :])) / scale
        u = v / scale
        n = u.norm().clamp_min(EPS)
        sc = self.c ** 0.5
        f = torch.tanh(sc * n) / (sc * n)
        fp = (1.0 / torch.cosh(sc * n) ** 2) / n - torch.tanh(sc * n) / (sc * n * n)
        dz_du = f * torch.eye(u.shape[0], device=u.device) + (fp / n) * torch.outer(u, u)
        dv_dh = (W2 * dg[None, :]) @ (W1 / self.sd[None, :])
        return (dz_du @ dv_dh) / scale

    # ---------------- pullback ----------------
    def _pullback(self, h, z_target):
        """iterated least-squares: find dh with E(h+dh) ~= z_target."""
        dh = torch.zeros_like(h)
        for _ in range(self.iters):
            hv = (h + dh).detach()
            z_now = self._z(hv)
            r = z_target - z_now
            if float(r.norm()) < 1e-4:
                break
            J = self._jac(hv)                                       # [k, d], closed form
            JJt = J @ J.T
            lam = self.eps * float(JJt.diagonal().mean())
            w = torch.linalg.solve(JJt + lam * torch.eye(JJt.shape[0], device=J.device), r)
            dh = dh + J.T @ w
        return dh

    def readings(self, H):
        """[K] chart readings of a batch of states (cheap: one MLP pass, no Jacobian)."""
        with torch.no_grad():
            z = self.ch.embed(self.ch.encode((H.float() - self.mu) / self.sd))
            return self.ch.busemann(z) if self.target == "ideal" else radius(z, self.kap)

    def __call__(self, H):
        """H [K, d] fp32 -> [K, d] displacements (already scaled; the caller multiplies by the dose weight w_t)."""
        out = []
        for h in H:
            h = h.detach().float()
            with torch.enable_grad():
                z = self._z(h)
                z_t = self._geodesic_target(z)
                dh = self._pullback(h, z_t)
            cap = self.rho_max * float(h.norm())
            n = float(dh.norm())
            if n > cap:
                dh = dh * (cap / max(n, EPS))
            with torch.no_grad():
                z1 = self._z(h + dh)
                self.log.append((float(self._reading(z)), float(self._reading(z1)),
                                 float(dist(z[None, :], z1[None, :], self.kap)), float(dh.norm() / h.norm())))
            out.append(dh)
        return torch.stack(out)
