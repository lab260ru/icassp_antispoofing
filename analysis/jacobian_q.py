#!/usr/bin/env python3
r"""The contraction premise, measured as an exact Jacobian singular value.

Assumption 2 of the paper ("$F$ is $q$-Lipschitz with $q<1$") has never been
measured. Two estimators have already failed and neither failure was about
tuning:

  * `src/common/boundaries.py` located repetition boundaries from text
    attention and read $q$ off the decay of boundary-state distances. The
    deep-layer attention centroid advances on ~51% of steps, so the located
    boundaries were noise. Self-defeating by construction: the flatter the
    attention over repeated spans (Lemma 1), the worse any attention-based
    localiser gets.
  * `analysis/contraction_probe.py` perturbed a single position by a finite
    delta and watched the difference propagate. Its own step-size linearity
    control failed, and the decisive repeated-vs-control contrast came out null
    (7/18 pairs, p=0.12).

This script is the designed successor, not a retry of either. It removes both
failure modes: boundaries are *constructed*, not *inferred*, and the derivative
is *exact*, not a difference quotient.

==============================================================================
1. What object is differentiated, and why it is the only honest choice
==============================================================================
The theorem's state $s_m$ is "whatever the decoder carries from repetition $m$
into repetition $m+1$". In a KV-cached autoregressive transformer that object is
NOT a single hidden vector. It is the set of per-layer key/value entries at the
token positions of repetition $m$; and $K,V$ at layer $\ell$ are (through
RMSNorm and a linear map) functions of layer $\ell$'s *input* at those
positions. So the decoder's own per-repetition state is

    S_m = ( h_\ell(t) : \ell = 0..L-1, t \in W_m )        [L x tau x d]

where $W_m$ is the token window of repetition $m$ and $h_\ell$ is the residual
stream entering layer $\ell$. Teacher-forcing the already-generated token
sequence fixes the tokens, and then $S_{m+1}$ is a deterministic differentiable
function $F$ of $S_m$ (everything outside $W_m \cup W_{m+1}$ held fixed). $F$ is
an endomorphism -- same shape in and out -- which is what makes a *contraction
factor* meaningful at all.

Why not the obvious thing, "hidden state at layer $\ell$, position $t_m$, mapped
to layer $\ell$, position $t_{m+1}$"? Because in a causal transformer that map
is **exactly zero**, not small. Information reaches position $t' > t$ only by
going *up* in depth through some layer's attention, so the Jacobian from depth
$\ell$ at $t$ to depth $\ell$ at $t'$ has no path at all. Any same-depth
single-vector "state" of a transformer is not a state. This is a structural
fact, and it is worth stating in the paper regardless of what $q$ comes out to
be: the theorem's $s_m$ has no single-layer realisation, only a whole-stack one.

The same argument gives a principled *per-layer* sweep. Fix $\ell$ and restrict
the state to depths $\ell..L-1$. The restriction is exactly closed: with tokens
fixed, depths $<\ell$ at positions in $W_{m+1}$ cannot depend on anything in
$W_m$ at depth $\ge \ell$. So $q_\ell := \sigma_{\max}$ of $F$ restricted to the
depth-$\ge\ell$ sub-stack is itself a legitimate contraction factor of a
legitimate state map, and $q_\ell$ is non-increasing in $\ell$ (smaller
invariant subspace). $q_0$ (whole stack) is the headline.

**Metric.** A singular value is metric-dependent, and residual-stream norms grow
by an order of magnitude with depth, so a raw Euclidean norm on the concatenated
stack would let the deepest layer decide the answer. We measure perturbations in
units of each layer's own RMS activation: coordinate $(\ell,t,i)$ is scaled by
$1/\mathrm{rms}_\ell$. This is not a cosmetic choice -- $K,V$ at layer $\ell$
are computed after RMSNorm, so $\delta/\mathrm{rms}_\ell$ is precisely the
perturbation size the rest of the network sees. The same scaling is used on the
input and the output, so $F$ stays an endomorphism of a fixed metric space.

==============================================================================
2. How boundary token indices are derived, and how far to trust them
==============================================================================
Teacher-forcing removes the need for the attention read-head that killed attempt
1, but it does not hand us the boundaries for free: the *text* spans of the $k$
repetitions are known exactly, while the *speech-token* spans they were rendered
into are not observed. We therefore construct them:

    T_gen  = number of generated speech tokens (recorded at generation time)
    tau    = round(T_gen / k)
    b_m    = plen + m * tau,   m = 0..k-1     (plen = prompt length in tokens)

i.e. a uniform partition of the rendered trajectory into $k$ equal spans. The
validations this admits are asserted, not assumed: exactly $k$ boundaries, spans
equal by construction, and the partition contained in the generated span.

What this does NOT establish is that span $m$ contains repetition $m$. It cannot
be established -- that is the same localisation problem that sank attempt 1, and
at $k\ge16$ the decoder frequently renders a number of repetitions different
from $k$, so even a perfect localiser would not put a boundary every $T/k$
tokens. The design answers this by not depending on it:

  * $\sigma_{\max}$ of the window map is defined for *any* window width; the
    boundary derivation only chooses which width to call "one repetition".
  * We therefore sweep the lag -- $\tau/2$, $\tau$, $2\tau$, and $\tau_{\rm
    rendered} = T_{gen}/c$ where $c$ is the CTC-judged rendered count -- and
    report $q$ as a function of it. A conclusion that survives the whole sweep
    does not rest on the partition being right.
  * $q(\tau)$ decreases with $\tau$, so the *shortest* plausible period gives
    the *largest* $q$: the least favourable reading for a "$q<1$" claim is the
    short-lag end, and it is reported.

==============================================================================
3. Two ways this measurement is biased, both stated before it is run
==============================================================================
(a) **Teacher-forcing makes $q$ a lower bound.** In real generation, perturbing
    repetition $m$'s state changes which tokens repetition $m+1$ emits, and that
    token channel is a large additional path from $S_m$ to $S_{m+1}$. Fixing the
    tokens deletes it. So the measured $q$ under-states the true per-repetition
    Lipschitz constant. Direction of the bias: *towards* the premise. A measured
    $q \ge 1$ would therefore be decisive against Assumption 2; a measured
    $q < 1$ is weaker evidence for it than it looks.
(b) **Independent per-layer perturbation over-states $\sigma_{\max}$.** We
    perturb each layer's cache entries independently, which is exactly what the
    cache permits but is a larger set of directions than the network's own
    forward pass can reach at a boundary. Direction of the bias: *against* the
    premise. (a) and (b) push opposite ways and we do not claim they cancel.

==============================================================================
4. PRE-COMMITTED INTERPRETATION  (written before any result was seen)
==============================================================================
Gate 0 -- the estimator must earn the right to be reported. Three self-tests,
all of which must pass or the numbers are not reportable and the script says so
in its verdict field:

  S1 synthetic positive control: the identical power-iteration code path,
     including the double-backward trick, applied to $y = A\tanh(x)+b$ whose
     Jacobian at $x=0$ is $A$, must recover $\sigma_{\max}(A)$ (known from a
     dense SVD) to better than 5%.
  S2 causality: the same estimator run *backwards* (state at $W_{m+1}$ -> state
     at $W_m$) must return exactly 0. A transformer cannot send information
     back in time; anything else means the hooks leak.
  S3 exactness: $\|Jv\|$ from the autograd path must agree with a finite
     difference $\|f(hv)-f(0)\|/h$ on the real model to better than 5%, at every
     step in a range spanning a factor of 10. This is the check attempt 2
     failed. Note the step must not be *too small*: $v$ is a unit vector spread
     over ~$2.5\times10^6$ coordinates, so at $h=10^{-3}$ each coordinate moves
     by ~$10^{-6}$ of an activation and the difference quotient is float32
     roundoff amplified by $1/h$. That is a fact about finite differences, not
     about the model, and it is the reason this script does not use them as an
     estimator. The recorded sweep includes the noise-dominated steps so the
     failure mode is visible in the results file rather than tuned away.

Headline quantity: $q_0$ at lag $=\tau$, whole stack, RMS metric, pooled over
repeated items, 95% CI by bootstrap over items.

  ONE STATISTIC WAS ADDED AFTER THE FIRST SELF-TEST, and it is flagged here
  rather than folded in silently. The S2 causality check prints a forward
  $\sigma_{\max}$ as a scale reference, and on the first item that number came
  out far above 1. Before running the sweep we therefore added a *descriptive*
  companion statistic, the typical-direction gain $\mathbb{E}\|Jv\|/\|v\|$ over
  random $v$ (a Frobenius-norm probe), because "$\sigma_{\max}>1$ while the
  average direction contracts hard" and "the map expands in every direction"
  are very different mechanistic pictures and only the second is a plain
  refutation. **The verdict logic below was not changed**: Assumption 2 is a
  uniform Lipschitz bound, so $\sigma_{\max}$ decides it, and the typical gain
  is reported alongside as description, never as the premise's test.

  PREMISE SUPPORTED  iff  the repeated arm's bootstrap CI for $q$ lies entirely
                     below 1.0  AND  the paired (repeated - control) difference
                     is significantly negative (Wilcoxon $p<0.05$ and a
                     bootstrap CI for the median difference excluding 0).
                     -> Track A is alive; the theorem's premise is measured.
  PREMISE DEAD       iff  the CI for $q$ includes or lies above 1.0, OR the
                     paired difference is not significantly negative
                     (repeated ~ control). The second disjunct matters as much
                     as the first: a contraction that is not specific to
                     repetition is not the theorem's mechanism, because the
                     control items count perfectly.
  INCONCLUSIVE       anything else (e.g. $q<1$ but the pairing is underpowered),
                     or any self-test failure.

If and only if the premise is supported, the implied horizon is computed from
the paper's Eq. (1), $N^\ast = \log(\mu/(2LC))/\log q$ with $C =
d(s_0,Fs_0)/(1-q)$, i.e.

    N*(q; rho) = log( rho (1-q) / 2 ) / log q,   rho := mu / (L * d(s_0,F s_0))

$\rho$ is the readout's decision margin expressed in units of one
repetition-step of state displacement, and it is not measurable here, so it is
declared rather than fitted: $\rho = 1$ ("the stop head must resolve one full
repetition-step of state change") is the headline, with $\rho \in \{0.1, 0.5\}$
reported alongside. The make-or-break comparison is against the observed
saturation of the repeated arm, $N^\ast_{\rm rep} = 23$ [16, 39].
For reference, under $\rho=1$: $q=0.80 \to N^\ast=10$, $q=0.90 \to 28$,
$q=0.95 \to 72$. So only $q \approx 0.88-0.91$ reproduces the observed horizon;
any $q$ far from that band means the premise, even if true, does not explain the
number the paper cares about, and the script says so.

Usage:
  python analysis/jacobian_q.py --selftest            # gate 0 only
  python analysis/jacobian_q.py --gpu 2               # full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HF_HOME", "/home/kirill/mnt/hdd_6tb_1/icassp_tts/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from src.common.gpus import DEFAULT_GPU, check_gpu  # noqa: E402
from common.registry import BY_KEY, DATA_ROOT  # noqa: E402
from models.llasa_gen import build_prompt, extract_speech_ids  # noqa: E402

# ---- pre-committed constants ------------------------------------------------
KS = (16, 24, 32)          # the repetition levels the task specifies
LAG_NAMES = ("half_tau", "tau", "two_tau", "tau_rendered")
SUBSTACK_FRACS = (0.0, 0.25, 0.5, 0.75)   # depth->= sweep, as fractions of L
N_ANCHORS = 3              # boundary pairs sampled per item
POWER_ITERS = 20
POWER_TOL = 2e-3
N_TYPICAL = 3              # random directions for the Frobenius/typical-gain probe
FD_STEPS = (1e-3, 1e-2, 0.3, 1.0, 3.0)   # S3 sweep; pass judged on >= 0.3 only
FD_PASS_STEPS = (0.3, 1.0, 3.0)
MAX_TOKENS = 2000          # truncation guard: fp32 math-SDPA + double backward
                           # peaks at 45 GB by 2100 tokens on a 49 GB card
RHOS = (1.0, 0.5, 0.1)
OBSERVED_NSTAR_REP = (16.0, 23.0, 39.0)   # horizon_ext.json: lo, point, hi


# ==============================================================================
# Exact Jacobian-vector products on a retained graph
# ==============================================================================
class JacobianOp:
    """Both directional derivatives of a fixed differentiable map, exactly.

    `y = f(x)` is built once and its graph retained; `Jv` uses the standard
    double-backward trick (the gradient of `J^T w` with respect to `w` is `J`,
    because `J^T w` is linear in `w`), and `JTu` is an ordinary VJP. Nothing
    here is a difference quotient, so there is no step size to choose and
    nothing for a linearity control to fail.
    """

    def __init__(self, y: torch.Tensor, x: torch.Tensor):
        self.y, self.x = y, x
        self.n_jv = 0
        self.n_jt = 0

    def Jv(self, v: torch.Tensor) -> torch.Tensor:
        w = torch.zeros_like(self.y, requires_grad=True)
        g = torch.autograd.grad(self.y, self.x, grad_outputs=w,
                                create_graph=True, retain_graph=True)[0]
        out = torch.autograd.grad(g, w, grad_outputs=v, retain_graph=True)[0]
        del g, w
        self.n_jv += 1
        return out.detach()

    def JTu(self, u: torch.Tensor) -> torch.Tensor:
        out = torch.autograd.grad(self.y, self.x, grad_outputs=u,
                                  retain_graph=True)[0]
        self.n_jt += 1
        return out.detach()


def typical_gain(op: JacobianOp, mask_in=None, mask_out=None,
                 n: int = N_TYPICAL, seed: int = 0) -> float:
    """`E ||Jv|| / ||v||` over isotropic random `v` -- i.e. `||J||_F/sqrt(dim)`.

    Descriptive only. The theorem needs a uniform bound (`sigma_max`); this says
    what happens to a direction nobody chose adversarially, which is what
    distinguishes "expands everywhere" from "contracts on average with a thin
    expanding subspace".
    """
    gen = torch.Generator(device=op.x.device).manual_seed(seed)
    vals = []
    for _ in range(n):
        v = torch.randn(op.x.shape, generator=gen, device=op.x.device,
                        dtype=op.x.dtype)
        if mask_in is not None:
            v = v * mask_in
        nv = float(v.norm())
        if nv == 0:
            continue
        u = op.Jv(v / nv)
        if mask_out is not None:
            u = u * mask_out
        vals.append(float(u.norm()))
    return float(np.mean(vals)) if vals else float("nan")


def top_singular_value(op: JacobianOp, mask_in=None, mask_out=None,
                       iters: int = POWER_ITERS, tol: float = POWER_TOL,
                       seed: int = 0) -> dict:
    """Power iteration on `J^T J`, using only exact directional derivatives.

    `mask_in`/`mask_out` project onto a sub-block, which is how the depth->=l
    sub-stack restriction is applied without rebuilding the graph.
    """
    gen = torch.Generator(device=op.x.device).manual_seed(seed)
    v = torch.randn(op.x.shape, generator=gen, device=op.x.device,
                    dtype=op.x.dtype)
    if mask_in is not None:
        v = v * mask_in
    nv = v.norm()
    if float(nv) == 0.0:
        return dict(sigma=0.0, converged=True, iters=0, history=[])
    v = v / nv
    hist: list[float] = []
    sigma = 0.0
    converged = False
    for i in range(iters):
        u = op.Jv(v)
        if mask_out is not None:
            u = u * mask_out
        sigma = float(u.norm())
        hist.append(sigma)
        if sigma == 0.0:
            converged = True
            break
        v_new = op.JTu(u / sigma)
        if mask_in is not None:
            v_new = v_new * mask_in
        n = float(v_new.norm())
        if n == 0.0:
            converged = True
            break
        v = v_new / n
        if i >= 2 and abs(hist[-1] - hist[-2]) <= tol * max(hist[-1], 1e-12):
            converged = True
            break
    return dict(sigma=sigma, converged=bool(converged), iters=len(hist),
                history=[float(h) for h in hist])


# ==============================================================================
# S1: synthetic positive control -- identical code path, known answer
# ==============================================================================
def selftest_synthetic(device: str, dim: int = 256, seed: int = 0) -> dict:
    """`y = A tanh(x) + b` has Jacobian `A` at `x = 0`. Recover sigma_max(A).

    Deliberately routed through `JacobianOp` and `top_singular_value` verbatim,
    including the double-backward, so a bug in the estimator cannot hide behind
    a simpler test harness. A tanh is in the path so the map is genuinely
    nonlinear and the test is of the *derivative at a point*, not of a linear
    algebra identity.
    """
    torch.manual_seed(seed)
    A = torch.randn(dim, dim, device=device, dtype=torch.float32) / dim ** 0.5
    # spread the spectrum so sigma_max is well separated (power iteration must
    # actually converge, not be rescued by a degenerate spectrum)
    U, S, Vh = torch.linalg.svd(A)
    S = torch.linspace(1.0, 0.05, dim, device=device) * 3.0
    A = (U * S) @ Vh
    b = torch.randn(dim, device=device)
    true = float(torch.linalg.svdvals(A)[0])

    x = torch.zeros(dim, device=device, requires_grad=True)
    y = A @ torch.tanh(x) + b
    op = JacobianOp(y, x)
    # tol 1e-5 rather than something tighter: the whole path is float32, whose
    # relative resolution is ~1e-7, so a tighter tolerance can never be met and
    # would report a converged iteration as unconverged.
    got = top_singular_value(op, iters=200, tol=1e-5, seed=seed)
    rel = abs(got["sigma"] - true) / true
    return dict(name="S1_synthetic_linear_map", true_sigma=true,
                measured_sigma=got["sigma"], rel_error=rel,
                iters=got["iters"], converged=got["converged"],
                passed=bool(rel < 0.05 and got["converged"]),
                note="power iteration on a known Jacobian; must be within 5%")


# ==============================================================================
# Teacher-forcing input reconstruction (copied from contraction_probe.py)
# ==============================================================================
def reconstruct_ids(tok, item: dict, sp_ids: np.ndarray) -> tuple[torch.Tensor, int]:
    """Rebuild [prompt ++ generated] ids from the recorded codec integers.

    Asserted, not assumed: mapping each integer back through `<|s_{id}|>` and
    re-running the SAME `extract_speech_ids` used at recording time must
    reproduce the saved array exactly.
    """
    prompt_ids = build_prompt(tok, item["text"])
    strs = [f"<|s_{int(i)}|>" for i in sp_ids]
    ids = tok.convert_tokens_to_ids(strs)
    if any(x is None for x in ids):
        raise ValueError(f"{item['item_id']}: unrecognised speech-token id")
    speech_t = torch.tensor(ids, dtype=prompt_ids.dtype).unsqueeze(0)
    full = torch.cat([prompt_ids, speech_t], dim=1)
    back = extract_speech_ids(tok.convert_ids_to_tokens(ids))
    if back != sp_ids.tolist():
        raise ValueError(f"{item['item_id']}: reconstruction round-trip mismatch")
    return full, int(prompt_ids.shape[1])


# ==============================================================================
# The window map S_m -> S_{m+1}
# ==============================================================================
class WindowMap:
    """Builds `S_{m+1} = F(S_m)` as a retained autograd graph.

    Injection is a `forward_pre_hook` on every decoder layer: layer l's input is
    increased by `rms_l * x[l]` on the columns of `W_in`. Readout captures every
    layer's input on the columns of `W_out`, divided by the same `rms_l`. The
    windows are disjoint, so the readout is untouched by the injection except
    through the network.

    Perturbing layer inputs is exactly perturbing the KV cache: K and V at layer
    l are functions of layer l's input at that position and nothing else.
    """

    def __init__(self, model, ids: torch.Tensor, w_in: tuple[int, int],
                 w_out: tuple[int, int], rms: torch.Tensor, requires_grad: bool = True,
                 x_value: torch.Tensor | None = None):
        L = model.config.num_hidden_layers
        d = model.config.hidden_size
        dev = ids.device
        n_in = w_in[1] - w_in[0]
        if x_value is None:
            self.x = torch.zeros(L, n_in, d, device=dev, dtype=torch.float32,
                                 requires_grad=requires_grad)
        else:
            self.x = x_value
        caps: dict[int, torch.Tensor] = {}
        xs = self.x

        def make(l: int):
            def hook(_mod, args):
                h = args[0]
                caps[l] = h[0, w_out[0]:w_out[1], :] / rms[l]
                pad = torch.zeros_like(h)
                pad[:, w_in[0]:w_in[1], :] = (rms[l] * xs[l]).unsqueeze(0)
                return (h + pad,) + tuple(args[1:])
            return hook

        handles = [model.model.layers[l].register_forward_pre_hook(make(l))
                   for l in range(L)]
        try:
            ctx = torch.enable_grad() if requires_grad else torch.no_grad()
            with ctx, sdpa_kernel([SDPBackend.MATH]):
                model.model(ids)
        finally:
            for h in handles:
                h.remove()
        self.y = torch.stack([caps[l] for l in range(L)], dim=0)  # [L, n_out, d]


def clean_states(model, ids: torch.Tensor, plen: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One clean pass: per-layer input RMS over the generated span, and the
    whole per-layer residual trajectory (used for the boundary-distance decay
    diagnostic and for the metric)."""
    L = model.config.num_hidden_layers
    caps: dict[int, torch.Tensor] = {}

    def make(l: int):
        def hook(_mod, args):
            caps[l] = args[0][0].detach()
            return None
        return hook

    handles = [model.model.layers[l].register_forward_pre_hook(make(l))
               for l in range(L)]
    try:
        with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
            model.model(ids)
    finally:
        for h in handles:
            h.remove()
    traj = torch.stack([caps[l] for l in range(L)], dim=0)  # [L, T, d]
    rms = traj[:, plen:, :].float().pow(2).mean(dim=(1, 2)).sqrt()  # [L]
    return rms.clamp_min(1e-8), traj


# ==============================================================================
# Boundary derivation (documented in the module docstring, validated here)
# ==============================================================================
def boundaries(plen: int, t_gen: int, k: int) -> tuple[list[int], int]:
    """Uniform partition of the rendered trajectory into k spans.

    Returns the k boundary token indices (absolute columns of the teacher-forced
    input) and the span width tau. Validation is asserted here rather than
    checked by a caller that might forget.
    """
    tau = int(round(t_gen / k))
    if tau < 8:
        return [], tau
    b = [plen + m * tau for m in range(k)]
    assert len(b) == k, "boundary count must equal k"
    spans = np.diff(np.array(b + [plen + k * tau]))
    assert spans.min() == spans.max() == tau, "spans must be equal by construction"
    assert b[0] >= plen and b[-1] < plen + t_gen, "partition must lie in the generated span"
    return b, tau


# ==============================================================================
# Per-item measurement
# ==============================================================================
def run_item(model, tok, device: str, item: dict, sp_ids: np.ndarray, seed: int,
             rendered_count: float | None, n_anchors: int, substacks: list[int],
             do_fd_check: bool = False) -> tuple[list[dict], dict]:
    full, plen = reconstruct_ids(tok, item, sp_ids)
    full = full.to(device)
    t_gen = int(len(sp_ids))
    k = int(item["k"])
    b, tau_k = boundaries(plen, t_gen, k)
    diag: dict = dict(item_id=item["item_id"], seed=seed, family=item["family"],
                      template=item["template"], k=k, t_gen=t_gen, plen=plen,
                      tau_k=tau_k, n_boundaries=len(b))
    if not b:
        diag["skipped"] = "tau too small"
        return [], diag

    rms, traj = clean_states(model, full, plen)
    diag["rms_by_layer"] = [float(v) for v in rms]

    # --- diagnostic: do boundary-state distances decay geometrically? --------
    # Free by-product of the clean pass and a direct check of Theorem 1(i) in
    # the same metric the Jacobian is measured in.
    S = []
    for m in range(k):
        lo, hi = b[m], b[m] + tau_k
        if hi > full.shape[1]:
            break
        S.append((traj[:, lo:hi, :].float() / rms.view(-1, 1, 1)))
    d_m = [float((S[m + 1] - S[m]).norm()) for m in range(len(S) - 1)]
    diag["boundary_distances"] = d_m
    del S, traj
    torch.cuda.empty_cache()

    # --- lag grid ------------------------------------------------------------
    lags = {"half_tau": max(8, tau_k // 2), "tau": tau_k, "two_tau": 2 * tau_k}
    if rendered_count and rendered_count >= 2:
        lags["tau_rendered"] = max(8, int(round(t_gen / rendered_count)))
    diag["lags"] = dict(lags)

    L = model.config.num_hidden_layers
    records: list[dict] = []
    fd_reports: list[dict] = []

    for lag_name, lag in lags.items():
        # anchors: window starts spread over the trajectory such that both
        # W_in = [a, a+lag) and W_out = [a+lag, a+2*lag) fit, and the truncated
        # input stays inside the memory guard.
        hi = plen + t_gen - 2 * lag
        if hi <= plen:
            continue
        cands = np.unique(np.linspace(plen, hi, n_anchors).astype(int))
        for a in cands.tolist():
            w_in = (a, a + lag)
            w_out = (a + lag, a + 2 * lag)
            t_eff = w_out[1]
            if t_eff > MAX_TOKENS:
                records.append(dict(item_id=item["item_id"], seed=seed,
                                    family=item["family"], template=item["template"],
                                    k=k, lag_name=lag_name, lag=lag, anchor=a - plen,
                                    substack=None, sigma=None,
                                    skipped="exceeds MAX_TOKENS"))
                continue
            ids = full[:, :t_eff]
            try:
                wm = WindowMap(model, ids, w_in, w_out, rms)
                op = JacobianOp(wm.y, wm.x)
                for l0 in substacks:
                    mask = torch.zeros(L, 1, 1, device=device)
                    mask[l0:] = 1.0
                    sd = hash((item["item_id"], seed, a, l0)) % (2 ** 31)
                    got = top_singular_value(op, mask_in=mask, mask_out=mask,
                                             seed=sd)
                    tg = typical_gain(op, mask_in=mask, mask_out=mask, seed=sd + 1)
                    records.append(dict(
                        item_id=item["item_id"], seed=seed, family=item["family"],
                        template=item["template"], k=k, lag_name=lag_name, lag=lag,
                        anchor=a - plen, substack=l0, sigma=got["sigma"],
                        typical_gain=tg,
                        converged=got["converged"], iters=got["iters"],
                        t_eff=t_eff, skipped=None))
                if do_fd_check and lag_name == "tau" and not fd_reports:
                    fd_reports.append(finite_difference_check(model, ids, w_in,
                                                              w_out, rms, op, wm))
                del op, wm
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not isinstance(e, torch.cuda.OutOfMemoryError) and \
                        "out of memory" not in str(e).lower():
                    raise
                records.append(dict(item_id=item["item_id"], seed=seed,
                                    family=item["family"], template=item["template"],
                                    k=k, lag_name=lag_name, lag=lag, anchor=a - plen,
                                    substack=None, sigma=None, skipped="OOM"))
            torch.cuda.empty_cache()
    if fd_reports:
        diag["fd_check"] = fd_reports[0]
    return records, diag


def finite_difference_check(model, ids, w_in, w_out, rms, op: JacobianOp,
                            wm: WindowMap) -> dict:
    """S3: the exact JVP against finite differences at a sweep of step sizes.

    The point is not that finite differences are a good estimator -- attempt 2
    showed they are not -- but that over the range of steps where a difference
    quotient is numerically meaningful at all, the two must agree; if they do
    not, the autograd path is wired to the wrong tensors. The two smallest
    steps are kept in the record precisely because they *fail*: they are the
    float32 roundoff wall that makes finite differences the wrong tool here,
    and they are excluded from the pass criterion by name, not silently.
    """
    gen = torch.Generator(device=wm.x.device).manual_seed(12345)
    v = torch.randn(wm.x.shape, generator=gen, device=wm.x.device)
    v = v / v.norm()
    jv = op.Jv(v)
    jn = float(jv.norm())
    sweep = []
    with torch.no_grad():
        base = WindowMap(model, ids, w_in, w_out, rms, requires_grad=False,
                         x_value=torch.zeros_like(v))
        for h in FD_STEPS:
            pert = WindowMap(model, ids, w_in, w_out, rms, requires_grad=False,
                             x_value=(h * v))
            fd = (pert.y - base.y) / h
            fn = float(fd.norm())
            cos = float((jv.flatten() @ fd.flatten()) / (jv.norm() * fd.norm() + 1e-12))
            sweep.append(dict(step=h, fd_norm=fn, cosine=cos,
                              rel_error=abs(jn - fn) / max(jn, 1e-12)))
            del pert, fd
    judged = [s for s in sweep if s["step"] in FD_PASS_STEPS]
    ok = bool(judged and all(s["rel_error"] < 0.05 and s["cosine"] > 0.99
                             for s in judged))
    return dict(name="S3_exact_vs_finite_difference", jvp_norm=jn, sweep=sweep,
                judged_steps=list(FD_PASS_STEPS), passed=ok,
                note="steps below 0.3 are float32 roundoff divided by h; "
                     "reported, not judged")


def causality_check(model, tok, device, item, sp_ids) -> dict:
    """S2: run the same estimator backwards in time. Must be exactly zero."""
    full, plen = reconstruct_ids(tok, item, sp_ids)
    full = full.to(device)
    t_gen = len(sp_ids)
    b, tau = boundaries(plen, t_gen, int(item["k"]))
    rms, _ = clean_states(model, full, plen)
    torch.cuda.empty_cache()
    a = b[1]
    lag = tau
    # forward map, for scale: W_in = [a, a+lag) -> W_out = [a+lag, a+2 lag)
    ids = full[:, :min(a + 2 * lag, MAX_TOKENS)]
    if a + 2 * lag > ids.shape[1]:
        a = max(plen, ids.shape[1] - 2 * lag)
    fwd = WindowMap(model, ids, (a, a + lag), (a + lag, a + 2 * lag), rms)
    s_fwd = top_singular_value(JacobianOp(fwd.y, fwd.x), iters=6, seed=1)["sigma"]
    del fwd
    torch.cuda.empty_cache()
    # backward map: inject LATER, read EARLIER
    bwd = WindowMap(model, ids, (a + lag, a + 2 * lag), (a, a + lag), rms)
    s_bwd = top_singular_value(JacobianOp(bwd.y, bwd.x), iters=6, seed=1)["sigma"]
    del bwd
    torch.cuda.empty_cache()
    return dict(name="S2_causality", sigma_forward=s_fwd, sigma_backward=s_bwd,
                passed=bool(s_bwd < 1e-9 and s_fwd > 1e-6),
                note="a causal transformer cannot move information back in time; "
                     "sigma_backward must be exactly 0 and sigma_forward must not be")


# ==============================================================================
# Aggregation, bootstrap, verdict
# ==============================================================================
def _boot_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 0,
             stat=np.median) -> tuple[float, float, float]:
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(vals, size=(n_boot, vals.size), replace=True), axis=1)
    return (float(stat(vals)), float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


def per_item(records: list[dict], lag_name: str, substack: int,
             field: str = "sigma") -> dict:
    """item_key -> median of `field` over anchors."""
    acc: dict[tuple, list[float]] = {}
    for r in records:
        if r.get("skipped") or r.get(field) is None:
            continue
        if r["lag_name"] != lag_name or r["substack"] != substack:
            continue
        key = (r["item_id"], r["seed"], r["family"], r["template"], r["k"])
        acc.setdefault(key, []).append(r[field])
    return {k: float(np.median(v)) for k, v in acc.items()}


def geometric_decay(diags: list[dict], family: str, n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """Theorem 1(i) read straight off the clean pass, in the same metric.

    Regress `log(d_m/d_0)` on `m` through the origin, pooled over items exactly
    as `analysis/pooled_q.py` does, where `d_m = ||S_{m+1} - S_m||` between
    consecutive uniform windows. This is the *observed* decay of boundary-state
    distances; it is a different object from the Jacobian (it is one realised
    trajectory, not a worst-case direction) and it is reported because a
    theorem that predicts `d_m ~ C q^m` should be checkable both ways.
    """
    curves = []
    for d in diags:
        if d.get("family") != family:
            continue
        c = np.asarray(d.get("boundary_distances") or [], dtype=float)
        if c.size >= 5 and np.all(np.isfinite(c)) and np.all(c > 0):
            curves.append(c)
    if len(curves) < 4:
        return dict(q=np.nan, lo=np.nan, hi=np.nan, n_items=len(curves), r2=np.nan)

    def slope(sel):
        x = np.concatenate([np.arange(c.size, dtype=float) for c in sel])
        y = np.concatenate([np.log(c / c[0]) for c in sel])
        return float((x @ y) / (x @ x)) if (x @ x) > 0 else np.nan

    sl = slope(curves)
    x = np.concatenate([np.arange(c.size, dtype=float) for c in curves])
    y = np.concatenate([np.log(c / c[0]) for c in curves])
    ss_res = float(((y - sl * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(seed)
    boots = [slope([curves[i] for i in rng.choice(len(curves), len(curves), True)])
             for _ in range(n_boot)]
    boots = [b for b in boots if np.isfinite(b)]
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(q=float(np.exp(sl)), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                n_items=len(curves),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan)


def nstar(q: float, rho: float) -> float:
    """Paper Eq. (1) with C = d(s_0, F s_0)/(1-q) and rho = mu/(L d(s_0,F s_0))."""
    if not np.isfinite(q) or q <= 0 or q >= 1:
        return float("nan")
    arg = rho * (1.0 - q) / 2.0
    if arg <= 0:
        return float("nan")
    return float(np.log(arg) / np.log(q))


def summarize(records: list[dict], pairs: list[tuple], selftests: dict,
              seed: int = 0, diags: list[dict] | None = None) -> dict:
    import scipy.stats as st

    out: dict = {"by_cell": {}, "by_cell_typical_gain": {}}
    lag_names = sorted({r["lag_name"] for r in records if not r.get("skipped")})
    substacks = sorted({r["substack"] for r in records
                        if r.get("substack") is not None})

    for field, dest in (("sigma", out["by_cell"]),
                        ("typical_gain", out["by_cell_typical_gain"])):
      for lag_name in lag_names:
        for l0 in substacks:
            pi = per_item(records, lag_name, l0, field=field)
            rep = np.array([v for k, v in pi.items() if k[2] == "word_rep"])
            ctl = np.array([v for k, v in pi.items() if k[2] == "control_word"])
            cell: dict = {}
            for nm, arr in (("repeated", rep), ("control", ctl)):
                m, lo, hi = _boot_ci(arr, seed=seed)
                cell[nm] = dict(n_items=int(arr.size), median=m, ci_lo=lo, ci_hi=hi,
                                mean=float(arr.mean()) if arr.size else np.nan,
                                frac_below_1=float((arr < 1).mean()) if arr.size else np.nan)
            # paired: same template/k/seed, repeated minus control
            diffs, ratios = [], []
            for wr_id, ct_id, s in pairs:
                a = next((v for k, v in pi.items() if k[0] == wr_id and k[1] == s), None)
                bb = next((v for k, v in pi.items() if k[0] == ct_id and k[1] == s), None)
                if a is None or bb is None:
                    continue
                diffs.append(a - bb)
                ratios.append(a / bb if bb > 1e-12 else np.nan)
            diffs = np.array(diffs, dtype=float)
            p = np.nan
            if diffs.size >= 6 and np.any(diffs != 0):
                try:
                    p = float(st.wilcoxon(diffs).pvalue)
                except Exception:  # noqa: BLE001
                    p = np.nan
            dm, dlo, dhi = _boot_ci(diffs, seed=seed)
            cell["paired"] = dict(
                n_pairs=int(diffs.size), median_diff=dm, ci_lo=dlo, ci_hi=dhi,
                wilcoxon_p=p,
                frac_repeated_lower=float((diffs < 0).mean()) if diffs.size else np.nan,
                median_ratio=float(np.nanmedian(ratios)) if ratios else np.nan)
            dest[f"{lag_name}|substack{l0}"] = cell

    # ---- headline: lag = tau, whole stack -----------------------------------
    head = out["by_cell"].get("tau|substack0", {})
    q_rep = head.get("repeated", {})
    pr = head.get("paired", {})
    st_pass = all(v.get("passed") for v in selftests.values() if isinstance(v, dict))

    q_hat = q_rep.get("median", np.nan)
    ci_lo, ci_hi = q_rep.get("ci_lo", np.nan), q_rep.get("ci_hi", np.nan)
    contracting = bool(np.isfinite(ci_hi) and ci_hi < 1.0)
    non_contracting = bool(np.isfinite(ci_lo) and ci_lo >= 1.0)
    sep = bool(np.isfinite(pr.get("wilcoxon_p", np.nan)) and pr["wilcoxon_p"] < 0.05
               and np.isfinite(pr.get("ci_hi", np.nan)) and pr["ci_hi"] < 0
               and np.isfinite(pr.get("median_diff", np.nan)) and pr["median_diff"] < 0)

    if not st_pass:
        verdict = "NOT REPORTABLE"
        why = ("a self-test failed; the estimator has not earned the right to "
               "report a number (see selftests)")
    elif contracting and sep:
        verdict = "PREMISE SUPPORTED"
        why = ("q < 1 on the repeated arm with the whole bootstrap CI below 1, "
               "and the repeated arm is significantly more contractive than its "
               "length-matched control")
    elif non_contracting:
        verdict = "PREMISE DEAD"
        why = ("the bootstrap CI for q on the repeated arm includes or exceeds "
               "1.0: the per-repetition map is not a contraction in this decoder")
    elif not sep:
        verdict = "PREMISE DEAD"
        why = ("whatever contraction is present is not specific to repetition -- "
               "the length-matched control, which the decoder counts correctly, "
               "is not measurably less contractive")
    else:
        verdict = "INCONCLUSIVE"
        why = "the two gates disagree or the estimate straddles 1.0"

    out["headline"] = dict(
        lag="tau", substack=0, metric="per-layer RMS",
        q_repeated=q_hat, ci=[ci_lo, ci_hi],
        q_control=head.get("control", {}).get("median", np.nan),
        control_ci=[head.get("control", {}).get("ci_lo", np.nan),
                    head.get("control", {}).get("ci_hi", np.nan)],
        paired=pr, contracting=contracting, repeated_below_control=sep,
        selftests_passed=st_pass, verdict=verdict, why=why)

    # least favourable reading: the largest q we measured anywhere on the
    # repeated arm across the lag grid at the whole-stack level.
    worst = -np.inf
    worst_cell = None
    for name, cell in out["by_cell"].items():
        if not name.endswith("substack0"):
            continue
        v = cell.get("repeated", {}).get("ci_hi", np.nan)
        if np.isfinite(v) and v > worst:
            worst, worst_cell = v, name
    out["least_favourable"] = dict(cell=worst_cell, q_repeated_ci_hi=float(worst)
                                   if np.isfinite(worst) else None)

    # implied horizon (reported regardless, flagged if the premise is not
    # supported, because "the number the theorem would imply" is informative
    # even when the premise fails)
    out["implied_nstar"] = {
        # N*(q) is increasing in q, so the CI on q maps straight through
        f"rho={r}": dict(point=nstar(q_hat, r), lo=nstar(ci_lo, r), hi=nstar(ci_hi, r))
        for r in RHOS}
    # Theorem 1(i) read off the clean trajectory, for comparison with the
    # Jacobian: the realised boundary-distance decay, pooled as in pooled_q.py.
    if diags:
        dec = {fam: geometric_decay(diags, fam)
               for fam in ("word_rep", "control_word")}
        # What horizon would the *decay* estimate imply, if one took it at face
        # value? Reported next to its own R^2 so nobody can quote the first
        # without the second.
        for fam, v in dec.items():
            v["implied_nstar_rho1"] = dict(
                point=nstar(v["q"], 1.0), lo=nstar(v["lo"], 1.0),
                hi=nstar(v["hi"], 1.0))
            v["usable"] = bool(np.isfinite(v.get("r2", np.nan)) and v["r2"] > 0.2)
        out["boundary_distance_decay"] = dec

    # The band of q that would reproduce the observed repeated-arm horizon,
    # so the measured value can be compared with what the theorem would need.
    grid = np.linspace(0.50, 0.999, 20000)
    ns = np.array([nstar(q, 1.0) for q in grid])
    band = {}
    for nm, target in zip(("lo", "point", "hi"), OBSERVED_NSTAR_REP):
        band[nm] = float(grid[int(np.nanargmin(np.abs(ns - target)))])
    out["q_band_reproducing_observed_nstar"] = dict(
        rho=1.0, q_for_nstar=band,
        note="q that Eq.(1) needs to place N* at the observed 16/23/39")

    out["observed_nstar_repeated"] = dict(lo=OBSERVED_NSTAR_REP[0],
                                          point=OBSERVED_NSTAR_REP[1],
                                          hi=OBSERVED_NSTAR_REP[2])
    lo_o, _, hi_o = OBSERVED_NSTAR_REP
    n_pt = out["implied_nstar"]["rho=1.0"]["point"]
    out["nstar_matches_observed"] = bool(np.isfinite(n_pt) and lo_o <= n_pt <= hi_o)
    return out


# ==============================================================================
# Item selection
# ==============================================================================
def load_meta(model: str) -> dict:
    path = Path(DATA_ROOT) / "tokens" / f"{model}_meta.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["item_id"], r["seed"])] = r
    return out


def rendered_counts(model: str) -> dict:
    import pandas as pd
    p = REPO / "data/results/behavioural_ctc.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    d = d[d.model == model]
    return {(r.item_id, int(r.seed)): float(r.count_final) for r in d.itertuples()}


def select(stim: dict, meta: dict, ks: tuple, seeds: tuple, bad_templates: list,
           max_pairs: int) -> list[tuple]:
    """(word_rep id, matched control id, seed), both members cap-free.

    `hit_cap` items were truncated by our own token budget, so `T_gen/k` is not
    a rendered period for them; requiring cap-free on both members keeps the two
    arms of a pair on identical footing. Judge-unmeasurable templates are
    dropped so the population matches `src/common/population.py`'s -- not
    because the state dynamics care, but because the rendered count we use for
    the `tau_rendered` lag comes from that judge.
    """
    pairs = []
    for iid, it in stim.items():
        if it["family"] != "word_rep" or it["k"] not in ks:
            continue
        if it["template"] in bad_templates:
            continue
        ctrl = next((c for c, v in stim.items() if v.get("control_of") == iid), None)
        if ctrl is None:
            continue
        for s in seeds:
            mw, mc = meta.get((iid, s)), meta.get((ctrl, s))
            if not mw or not mc or mw.get("hit_cap") or mc.get("hit_cap"):
                continue
            pairs.append((iid, ctrl, s))
    pairs.sort()
    return pairs[:max_pairs] if max_pairs else pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llasa1b")
    ap.add_argument("--gpu", type=int, default=DEFAULT_GPU)
    ap.add_argument("--stimuli", default=str(REPO / "data/stimuli/stimuli.jsonl"))
    ap.add_argument("--out", default=str(REPO / "data/results/jacobian_q.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--n-anchors", type=int, default=N_ANCHORS)
    ap.add_argument("--selftest", action="store_true",
                    help="run gate 0 (the three self-tests) and exit")
    ap.add_argument("--resummarize", default="",
                    help="path to an existing results JSON: recompute the "
                         "summary block from its stored records and rewrite it, "
                         "no GPU. Use after changing an aggregation, so the "
                         "results file always matches the current script.")
    args = ap.parse_args()

    if args.resummarize:
        path = Path(args.resummarize)
        res = json.loads(path.read_text())
        pairs = [tuple(x) for x in res["config"].get("pairs", [])]
        if not pairs:
            # results written before `pairs` was stored: rebuild the identical
            # selection from the same inputs rather than guessing it from the
            # records, so the paired test is over exactly the same pairs.
            stim = {json.loads(l)["item_id"]: json.loads(l)
                    for l in open(args.stimuli)}
            from src.common.population import excluded_templates
            pairs = select(stim, load_meta(res["config"]["model"]),
                           tuple(res["config"]["ks"]), tuple(res["config"]["seeds"]),
                           excluded_templates(), 0)
        res["summary"] = summarize(res["records"], pairs, res["selftests"],
                                   diags=res.get("diagnostics"))
        path.write_text(json.dumps(res, indent=2, default=lambda x: None))
        h = res["summary"]["headline"]
        print(f"re-summarised {path}\nVERDICT: {h['verdict']} -- {h['why']}")
        return

    check_gpu(args.gpu)

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    spec = BY_KEY[args.model]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[{spec.key}] loading {spec.hf_id} float32 on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers
    substacks = sorted({int(round(f * L)) for f in SUBSTACK_FRACS})
    print(f"[{spec.key}] L={L} d={model.config.hidden_size} substacks={substacks}",
          flush=True)

    stim = {json.loads(l)["item_id"]: json.loads(l) for l in open(args.stimuli)}
    meta = load_meta(args.model)
    from src.common.population import excluded_templates
    bad = excluded_templates()
    pairs = select(stim, meta, tuple(args.ks), tuple(args.seeds), bad, args.max_pairs)
    print(f"[{spec.key}] {len(pairs)} pairs (k in {args.ks}, seeds {args.seeds}, "
          f"cap-free both members, templates excluded: {bad})", flush=True)
    if not pairs:
        raise SystemExit("no eligible pairs")

    tok_dir = Path(DATA_ROOT) / "tokens" / args.model
    counts = rendered_counts(args.model)

    # ---------------- gate 0: the self-tests --------------------------------
    print("\n=== gate 0: self-tests ===", flush=True)
    selftests: dict = {}
    s1 = selftest_synthetic(device)
    selftests["S1"] = s1
    print(f"  S1 synthetic: true={s1['true_sigma']:.6f} got={s1['measured_sigma']:.6f} "
          f"rel={s1['rel_error']:.2e} -> {'PASS' if s1['passed'] else 'FAIL'}", flush=True)

    wr0, ct0, s0 = pairs[0]
    sp0 = np.load(tok_dir / f"{wr0}_s{s0}.npy")
    s2 = causality_check(model, tok, device, stim[wr0], sp0)
    selftests["S2"] = s2
    print(f"  S2 causality: forward={s2['sigma_forward']:.6e} "
          f"backward={s2['sigma_backward']:.6e} -> "
          f"{'PASS' if s2['passed'] else 'FAIL'}", flush=True)

    if args.selftest:
        # S3 needs one real item's graph; run it on the first pair only.
        recs, diag = run_item(model, tok, device, stim[wr0], sp0, s0,
                              counts.get((wr0, s0)), 1, [0], do_fd_check=True)
        s3 = diag.get("fd_check")
        if s3:
            selftests["S3"] = s3
            for row in s3['sweep']:
                print(f"     h={row['step']:<8g} ||fd||={row['fd_norm']:.5f} "
                      f"rel={row['rel_error']:.2e} cos={row['cosine']:.6f}"
                      f"{'   [judged]' if row['step'] in s3['judged_steps'] else ''}")
            print(f"  S3 exact-vs-FD: ||Jv||={s3['jvp_norm']:.5f} -> "
                  f"{'PASS' if s3['passed'] else 'FAIL'}", flush=True)
        ok = all(v["passed"] for v in selftests.values())
        print(f"\ngate 0: {'PASS' if ok else 'FAIL'}")
        return

    # ---------------- the sweep ---------------------------------------------
    items: list[tuple[str, int]] = []
    for wr, ct, s in pairs:
        items += [(wr, s), (ct, s)]
    seen = set()
    items = [i for i in items if not (i in seen or seen.add(i))]

    all_records: list[dict] = []
    diags: list[dict] = []
    t0 = time.time()
    for n, (iid, s) in enumerate(items):
        f = tok_dir / f"{iid}_s{s}.npy"
        if not f.exists():
            print(f"  [{iid}/s{s}] tokens missing", flush=True)
            continue
        sp = np.load(f)
        try:
            recs, diag = run_item(model, tok, device, stim[iid], sp, s,
                                  counts.get((iid, s)), args.n_anchors, substacks,
                                  do_fd_check=("S3" not in selftests))
        except Exception as e:  # noqa: BLE001
            print(f"  [{iid}/s{s}] FAILED: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if "S3" not in selftests and diag.get("fd_check"):
            s3 = diag["fd_check"]
            selftests["S3"] = s3
            for row in s3['sweep']:
                print(f"     h={row['step']:<8g} ||fd||={row['fd_norm']:.5f} "
                      f"rel={row['rel_error']:.2e} cos={row['cosine']:.6f}"
                      f"{'   [judged]' if row['step'] in s3['judged_steps'] else ''}")
            print(f"  S3 exact-vs-FD: ||Jv||={s3['jvp_norm']:.5f} -> "
                  f"{'PASS' if s3['passed'] else 'FAIL'}", flush=True)
        all_records += recs
        diags.append(diag)
        el = time.time() - t0
        print(f"[{n+1}/{len(items)}] {iid} s{s} {diag['family'][:4]} k={diag['k']} "
              f"tau={diag['tau_k']} recs={len(recs)} {el/(n+1):.1f}s/item", flush=True)
        torch.cuda.empty_cache()

    ok_records = [r for r in all_records if not r.get("skipped")]
    print(f"\n{len(ok_records)} usable measurements "
          f"({len(all_records) - len(ok_records)} skipped)", flush=True)
    n_unconv = sum(1 for r in ok_records if not r.get("converged"))
    print(f"power iteration did not converge in {n_unconv}/{len(ok_records)} cells "
          f"(tol {POWER_TOL}, max {POWER_ITERS} iters)", flush=True)

    summary = summarize(all_records, pairs, selftests, diags=diags)

    print("\n--- q by lag and sub-stack (median over items, 95% bootstrap CI) ---")
    print(f"{'cell':>26s} {'repeated':>22s} {'control':>22s} {'paired p':>9s}")
    for name in sorted(summary["by_cell"]):
        c = summary["by_cell"][name]
        r, ct = c["repeated"], c["control"]
        print(f"{name:>26s} {r['median']:7.4f} [{r['ci_lo']:6.4f},{r['ci_hi']:6.4f}] "
              f"{ct['median']:7.4f} [{ct['ci_lo']:6.4f},{ct['ci_hi']:6.4f}] "
              f"{c['paired']['wilcoxon_p']:9.4f}")

    h = summary["headline"]
    print(f"\nHEADLINE  q_repeated = {h['q_repeated']:.4f} "
          f"[{h['ci'][0]:.4f}, {h['ci'][1]:.4f}]   "
          f"q_control = {h['q_control']:.4f}")
    print(f"least favourable reading anywhere on the lag grid: "
          f"{summary['least_favourable']}")
    print(f"\nVERDICT: {h['verdict']} -- {h['why']}")
    print(f"implied N* (rho=1): {summary['implied_nstar']['rho=1.0']}")
    print(f"observed N*_rep: {summary['observed_nstar_repeated']}  "
          f"match={summary['nstar_matches_observed']}")

    result = dict(
        config=dict(model=args.model, hf_id=spec.hf_id, dtype="float32",
                    gpu=args.gpu, n_layers=L, hidden=model.config.hidden_size,
                    ks=args.ks, seeds=args.seeds, n_pairs=len(pairs),
                    pairs=[list(p) for p in pairs],
                    n_items=len(items), substacks=substacks,
                    power_iters=POWER_ITERS, power_tol=POWER_TOL,
                    max_tokens=MAX_TOKENS, n_anchors=args.n_anchors,
                    metric="per-layer RMS-normalised residual stack",
                    lag_grid=list(LAG_NAMES)),
        selftests=selftests, summary=summary, diagnostics=diags,
        records=all_records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=lambda x: None))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
