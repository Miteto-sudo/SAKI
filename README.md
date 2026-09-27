# SAKI: Core Reference Implementation

This release contains the framework-independent core implementation of SAKI. It exposes the mathematical bridge, exact maximal coupling, first-rejection block semantics, and coupling-routed training objective in a compact package that can be inspected and tested on CPU.

This is the intended partial/reference-code boundary, not the complete VERL/SGLang/Ray distributed training system. The omitted components are distributed and systems integrations rather than changes to the trust-region bridge, maximal-coupling distribution, or routed objective. This package alone does not reproduce the paper's end-to-end systems throughput result.

## Installation

Use Python 3.10 or newer and install a PyTorch build compatible with your CUDA runtime. Then install this package without replacing that PyTorch build:

```bash
python -m pip install -e . --no-deps
```

The reference tests and example also run on CPU.

## Quick check

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/run_tests.sh
CUDA_VISIBLE_DEVICES="" python -X faulthandler examples/toy_rollout.py
```

## Method semantics

At each prefix, the frozen student proposes a token from $p$. The teacher and student define the full-vocabulary geometric bridge

$$
q_\beta(v) \propto p(v)^{1-\beta}T(v)^\beta,
\qquad D_{\mathrm{KL}}(q_\beta\|p) \leq \epsilon.
$$

Maximal coupling retains a proposal $z$ with probability

$$
\min\left(1, \frac{q(z)}{p(z)}\right).
$$

If the proposal is rejected, the rollout token is sampled from the positive residual $[q-p]_+$. Verification commits the accepted prefix plus the first residual correction and invalidates the speculative suffix. The correction token changes the subsequent rollout prefix, but it is not the direct supervision target.

The realized coupling event routes training:

- accepted position: sampled-token K1/RKL policy-gradient surrogate;
- correction position: the K1 contribution is exactly zero and is replaced by teacher-mode NLL, $-\log \pi_\theta(\arg\max_v T(v))$;
- the correction coefficient is 1.0;
- both branches are summed and jointly normalized by the total number of valid response tokens.

The exact $\epsilon=0$ endpoint returns $\beta=0$, $q=p$, and zero corrections.

## Training implementation

At accepted positions, the detached sampled-token K1/RKL signal is optimized with the training framework's one-epoch importance-ratio surrogate. The reference configuration uses a clip ratio of 0.2, a dual-clip ratio of 3.0, and clips per-token K1 values to `[-10, 10]`.

At correction positions, the accepted-token K1 contribution is fully masked out and replaced by teacher-mode NLL against the teacher Top-1 token with coefficient 1.0. Both branches are jointly normalized by the total number of valid response tokens. The residual correction token determines the subsequent rollout prefix and can differ from the teacher-mode supervision target.

## Paper-to-code map

- trust-region geometric bridge: `saki_opd/math/bridge.py`
- maximal coupling and positive-residual sampling: `saki_opd/math/coupling.py` and `saki_opd/math/residual.py`
- exact-\(q\) first-rejection block verification: `saki_opd/engine/q_block.py`
- coupling-routed K1 / teacher-mode objective: `saki_opd/objectives/coupling_routed.py`
- epsilon schedules and rollout validation: `saki_opd/runtime/validation.py`

The block verifier returns compact, token-aligned metadata including committed tokens, correction events, selected log-probabilities, bridge diagnostics, and teacher-mode targets. Values beyond the committed prefix are masked invalid.

## Main configuration

`configs/saki_main.yaml` records the reference paper configuration:

- speculative block size: 8;
- epsilon: 0.02 to exactly 0.0 over steps 1--51;
- correction objective: teacher Top-1;
- correction coefficient: 1.0;
- normalization: all valid response tokens;
- accepted-position optimizer: one-epoch clipped PPO surrogate for sampled-token K1/RKL.

## Release boundary

See `RELEASE_SCOPE.md` for the exact included and omitted components.

## License

This repository provides the source code associated with the SAKI paper. No open-source license is currently granted.
