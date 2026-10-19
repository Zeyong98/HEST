"""Hyperbolic probe ("chart"): encoder MLP -> Poincare ball, with a learnable curvature and two readouts.

Geometry (kappa-stereographic model, kappa < 0 = hyperbolic):
    mobius_add   Mobius addition
    dist         geodesic distance
    exp0         exponential map at the origin
    radius       hyperbolic distance to the origin (the "radius" readout)
GeoHead   encoder d_in -> 1024 -> d_out, exp0 to the ball, learnable kappa (softplus), running RMS scale
ChartV3   GeoHead + the Busemann readout b_xi(z) = log(||xi - y||^2 / (1 - ||y||^2)), y = z sqrt|kappa|,
          with a learnable ideal point xi (the "Busemann" readout)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DIM = 128
HID = 1024
EPS = 1e-6


def mobius_add(x, y, kappa):
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 - 2 * kappa * xy - kappa * y2) * x + (1 + kappa * x2) * y
    den = 1 - 2 * kappa * xy + kappa ** 2 * x2 * y2
    return num / den.clamp_min(EPS)


def kval(kappa):
    return float(kappa.detach() if torch.is_tensor(kappa) else kappa)


def dist(x, y, kappa):
    if kval(kappa) == 0.0:
        return 2.0 * (x - y).norm(dim=-1)
    diff = mobius_add(-x, y, kappa)
    dn = diff.norm(dim=-1)
    if kval(kappa) < 0:
        sc = (-kappa).clamp_min(EPS).sqrt()
        return (2.0 / sc) * torch.atanh((sc * dn).clamp(max=1 - 1e-6))
    sk = kappa.clamp_min(EPS).sqrt()
    return (2.0 / sk) * torch.atan(sk * dn)


def exp0(v, kappa):
    if kval(kappa) == 0.0:
        return v
    vn = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    if kval(kappa) < 0:
        sc = (-kappa).clamp_min(EPS).sqrt()
        z = torch.tanh(sc * vn) * v / (sc * vn)
        maxn = (1 - 1e-5) / sc
        zn = z.norm(dim=-1, keepdim=True)
        return torch.where(zn > maxn, z * maxn / zn, z)
    sk = kappa.clamp_min(EPS).sqrt()
    arg = (sk * vn).clamp(max=math.pi / 2 - 0.05)
    return torch.tan(arg) * v / (sk * vn)


def radius(z, kappa):
    if kval(kappa) == 0.0:
        return 2.0 * z.norm(dim=-1)
    zn = z.norm(dim=-1)
    if kval(kappa) < 0:
        sc = (-kappa).clamp_min(EPS).sqrt()
        return (2.0 / sc) * torch.atanh((sc * zn).clamp(max=1 - 1e-6))
    sk = kappa.clamp_min(EPS).sqrt()
    return (2.0 / sk) * torch.atan(sk * zn)


class GeoHead(nn.Module):
    def __init__(self, geometry, d_in=4096, hid=HID, d_out=N_DIM, kappa_init=1.0):
        super().__init__()
        self.geometry = geometry
        self.mlp = nn.Sequential(nn.Linear(d_in, hid), nn.GELU(), nn.Linear(hid, d_out))
        a0 = math.log(math.exp(kappa_init) - 1.0)
        self.raw_kappa = nn.Parameter(torch.tensor(a0))
        self.register_buffer("scale", torch.tensor(1.0))

    def kappa(self):
        if self.geometry == "euclid":
            return torch.tensor(0.0, device=self.raw_kappa.device)
        k = F.softplus(self.raw_kappa).clamp(1e-3, 4.0)
        return -k if self.geometry == "hyp" else k

    def encode(self, x):
        return self.mlp(x)

    def embed(self, v, rms=None):
        s = self.scale if rms is None else rms
        return exp0(v / s.clamp_min(EPS), self.kappa())


class ChartV3(GeoHead):
    def __init__(self, geometry, score, **kw):
        super().__init__(geometry, **kw)
        self.score_type = score
        g = torch.Generator().manual_seed(7)
        self.raw_xi = nn.Parameter(torch.randn(kw.get("d_out", 128), generator=g))

    def xi(self):
        return self.raw_xi / self.raw_xi.norm().clamp_min(EPS)

    def busemann(self, z):
        xi = self.xi()
        kap = self.kappa()
        if float(kap.detach()) == 0.0:
            return -(z * xi).sum(-1)
        sc = (-kap).clamp_min(EPS).sqrt()
        y = z * sc
        num = ((xi - y) ** 2).sum(-1).clamp_min(EPS)
        den = (1.0 - (y * y).sum(-1)).clamp_min(EPS)
        return torch.log(num / den)

    def score(self, z):
        if self.score_type == "busemann":
            return self.busemann(z)
        return radius(z, self.kappa())

    def score_from_x(self, x, rms=None):
        return self.score(self.embed(self.encode(x), rms))
