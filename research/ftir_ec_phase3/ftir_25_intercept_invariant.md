# ftir_25 — the intercept as an absorption target, and what is invariant about it

Restates the Addis intercept as a quantity in Fabs units, and reports how it behaves
across the six setup-matrix calibrations. Everything here derives from numbers already
committed in `ftir_13` and `ftir_19`; no spectra, no Drive access, and no refit are
required, which is why it is written up ahead of the analyses in
`INTERCEPT_ATTACK_PLAN.md` that do need data.

Numbering note: the attack plan pencils in "ftir_25" for the BrC subtract-and-recrossplot
notebook (its item 5). That work should take the next free number instead.

## The quantity

The Addis crossplots fit `y = a·x + b` with `y` = FTIR-predicted EC (µg/m³) and
`x` = Fabs / MAC (µg/m³). Setting `y = 0` gives the Fabs value at which the calibration
predicts no EC at all:

```
C = |b| · MAC / a        [Mm⁻¹]
```

With `b` negative, `C > 0`: a constant absorption excess carried by Fabs with no FTIR-EC
behind it. This is the intercept restated as an absorption target — "find or rule out
C Mm⁻¹ of constant non-EC absorption at the HIPS wavelength (~633 nm)" — rather than as a
regression artifact.

## C does not depend on the MAC fork

`ftir_19` establishes two identities across all six setups: the intercept is MAC-invariant,
and `a@MAC 6 = 0.6 × a@MAC 10`. Together these make C invariant as well:

```
|b| · 6 / (0.6 · a)  ≡  |b| · 10 / a
```

Confirmed numerically (intercept and slopes from `ftir_19` cell 5; C computed here):

| Setup | intercept | a @ MAC 10 | a @ MAC 6 | **C @ MAC 10** | **C @ MAC 6** |
|---|---:|---:|---:|---:|---:|
| Deployed SPARTAN | −4.17 | 1.90 | 1.14 | 21.95 | 21.95 |
| Biomass-smoke (906) | −6.91 | 2.65 | 1.59 | 26.08 | 26.08 |
| Ethiopia-shaped smoke (300) | −3.69 | 1.75 | 1.05 | 21.09 | 21.09 |
| Spectral analogs (400) | −6.43 | 2.91 | 1.75 | 22.10 | 22.05 |
| Lowest-OC/EC (800) | −3.22 | 1.59 | 0.95 | 20.25 | 20.34 |
| Lowest-OC/EC + AIRSpec | −1.62 | 0.86 | 0.51 | 18.84 | 19.06 |

(Residual differences in the last two rows are 2-dp rounding in the published slopes. On
the 6-dp values in `ftir_13` cell 11 the agreement is exact: lowest-OC/EC raw gives
20.32 Mm⁻¹ at either MAC, +AIRSpec 18.85, deployed 21.97, smoke-906 +AIRSpec 18.15.)

**So the target is well-posed whether or not the MAC question is ever settled.** This is
worth stating plainly, because the natural first phrasing — "both branches of the MAC fork
imply the same ~20 Mm⁻¹" — reads as two independent estimates agreeing when it is an
algebraic identity. The two figures usually quoted (≈19 and ≈20 Mm⁻¹) differ because they
come from two different *models*, raw lowest-OC/EC and its AIRSpec-corrected counterpart;
each on its own returns the same C at both MACs.

## C is stable where slope and intercept are not

Across the six setups, slope spans **3.4×** (0.86–2.91) and intercept **4.3×**
(−1.62 to −6.91). C spans **1.4×**:

```
C = 18.8 – 26.1 Mm⁻¹,  median 21.5 Mm⁻¹
```

Median Addis Fabs is 47.11 Mm⁻¹ (`ftir_13` cell 13), so the implied EC-free absorption is
**≈ 46% of a typical Addis filter's**.

This is the premise behind the attack plan's item 2 — "if every setup recovers the same
offset, the offset lives in the Addis x-data rather than in any one model" — evaluated
without refitting anything.

## Why that is weaker evidence than it looks

C is invariant to any *multiplicative* rescaling of the predicted-EC axis. If one model's
predictions differ from another's by a factor k, then both the mean prediction and the
slope scale by k, so `ȳ/a` — and therefore C — is unchanged. Changing MAC is exactly such
a rescaling, which is the identity above; changing training composition is largely one too.

So the six setups are not six independent routes converging on 20 Mm⁻¹. C is the component
of the disagreement that cohort choice and MAC choice *cannot* move, and its stability says
the additive part of the discrepancy is robust to every scaling decision made so far. That
is a real and useful statement. It is not corroboration by independent measurement, and it
should not be presented as such.

The corollary for item 2 of the attack plan: refitting as `y = a·(x − c)` is a
reparameterization, not a new fit — `c = |b|/a` identically. It is still worth doing for
one reason, which is to put a bootstrap confidence interval directly on `c` rather than
propagating one from `a` and `b`.

## What this does and does not license

- The intercept can be quoted as an absorption target with a magnitude and a wavelength,
  independent of the MAC fork.
- The target is large: ~40–45% of median Addis Fabs, not a marginal correction.
- Candidate owners (brown carbon, dust, a HIPS-generic offset, an FTIR zero error) are
  **not** discriminated by anything here. C is a restatement of the intercept, not new
  evidence about its cause; the Tier-1 and Tier-2 analyses in `INTERCEPT_ATTACK_PLAN.md`
  are what separate them.
- In particular, nothing here shows the absorption is real rather than an artifact of the
  FTIR side reading low. Item 8 of the plan (blanks and low-EC behaviour) remains the cheap
  loophole-closer.

## Reproducing the table

```bash
uv run python research/ftir_ec_phase3/scripts/run_ftir_25_intercept_invariant.py
```

Inputs are the committed constants above; the script recomputes C at both MACs, re-checks
the MAC-invariance identity, and prints the spread.
