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
reparameterization, not a new fit — `c = |b|/a` identically, confirmed numerically to
1×10⁻¹⁰ against a nonlinear least-squares fit, and even the standard error agrees
(asymptotic SE(c) = 0.1173 from the NLS Jacobian, 0.1173 by the delta method from the
existing fit). It is still worth doing for one reason, which is to put a bootstrap
confidence interval directly on `c` — and `ftir_15`'s committed `addis_bootstrap_draws.csv`
already carries per-draw `slope` and `intercept`, so `c_b = −b_b/a_b` per draw gives that CI
with no refitting at all.

## The sharpest available test of "the offset is in x" — and it passes

AIRSpec baselining changes only the **y-side**: it transforms the spectra the PLS model is
built from and evaluated on, and never touches Fabs. It moves the intercept from −3.2215 to
−1.6151, a halving. That looks at first like a refutation — an artifact living in the
x-data should not be halvable by a y-side transform.

It is not a refutation, because the same transform also halves the slope, and under the
x-side model `y = a·(x − c)` the intercept **is** `−a·c`. A y-side gain change therefore
*must* move the intercept in proportion to the slope, leaving `c` fixed. The numbers behave
exactly that way:

| | raw | +AIRSpec | ratio |
|---|---:|---:|---:|
| slope `a` | 1.585381 | 0.857004 | 0.5406 |
| intercept `b` | −3.221502 | −1.615099 | 0.5013 |
| **`c = −b/a`** | **2.0320** | **1.8846** | **0.9274** |

The intercept ratio (0.5013) and the slope ratio (0.5406) agree to within 8%; a pure gain
change predicts −1.7414 against the observed −1.6151, so the slope move accounts for
essentially all of the halving. `c` shifts by −7.3% where `a` and `b` each move ~50%.

So the x-side premise survives the one test in this repo that could have killed it cheaply.
What remains is that residual −7.3%: `c` is not *perfectly* invariant, and its drift is
systematic rather than noisy — across the six setups `c` runs 1.885–2.608 (CV 11.2%), and
the best-performing setup has the lowest `c`. A single shared `c` fitted jointly across all
six (7 parameters against 12) costs only **1.6%** in total RSS, ĉ = 2.2634 µg/m³
(≈ 22.6 Mm⁻¹) — nearly free, but confounded by the same gain-invariance, so it corroborates
rather than proves.

## What would actually discriminate

Since `c` is blind to gain, no amount of refitting on Fabs-derived quantities can settle
where the offset lives. That needs an EC reference on the same Addis filters that is not
derived from Fabs. Two candidates in the committed dataset, and the attack plan names the
wrong one as the hazard:

| column | n | R² vs Fabs/10 | vs `EC_ftir` | verdict |
|---|---:|---:|---:|---|
| `ChemSpec_BC_PM2.5` | 188 | **0.9982** | — | **x-circular.** Fabs/10 rounded to 2 dp: implied MAC median 10.0003 (IQR 9.99–10.01), 86.7% of filters within 0.005. |
| `ChemSpec_EC_PM2.5` | 175 | 0.7904 | **r² = 0.999693** | **y-circular.** Ratio to `EC_ftir` median 1.0000 (IQR 0.999–1.001), median absolute difference 0.0030 — the 2-dp rounding half-width. It *is* the FTIR-EC product, routed through the speciation table. |

**Neither column can arbitrate, and the second one nearly fooled this note.** An earlier
revision proposed `ChemSpec_EC` as the independent reference on the strength of its *not*
being a Fabs transform. That inference was wrong: not-x-circular does not imply
independent, and this column is circular on the other axis. The tell was visible and
misread — its R² of 0.7904 against Fabs matches `EC_ftir`'s own 0.76 because it *is*
`EC_ftir`. `docs/open-items.md` has carried the warning (r² = 0.99992 in the four-sites
data); it reproduces at ETAD.

Two join traps worth keeping regardless, since both fail silently: each ChemSpec filter
carries a second ~0.07 µg/m³ floor row, so averaging halves the values and doubles any
implied MAC; and `config.BASE_FILTER_ID_PATTERN` matches only the suffixed form
(`ETAD-0001-1`), returning NaN for the unsuffixed ChemSpec ids and emptying the join
entirely.

### The consequence: nothing in hand can settle this

There is **no EC reference in the committed dataset that is independent of both axes**.
Every candidate is circular with Fabs or with FTIR. Taken with the other results now on the
table — IMPROVE HIPS runs through the origin, so the offset is not a generic instrument
zero; the FTIR axis is exonerated against MA350 BC(880); and the MA350 cannot measure a BrC
share at all — the three surviving explanations for the ~21 Mm⁻¹ are a **loading-dependent
HIPS artifact**, **curve geometry** (a straight line through a concave Fabs-vs-EC
relationship manufactures a negative intercept with no offset present), and **real non-EC
absorption**. Nothing in hand discriminates them.

That is what makes the quartz-TOR campaign (`ftir_16`: 11–13 days per season, ~36 filters
in total, quartz only) the decisive measurement rather than one option among several. It is
the only route to an EC reference that is circular with neither axis.

Joining on `FilterId` also needs care: HIPS rows carry the replicate suffix
(`ETAD-0001-1`) while ChemSpec rows do not (`ETAD-0001`), and
`config.BASE_FILTER_ID_PATTERN` only matches the suffixed form — it returns NaN for an
already-base id, so a naive `str.extract` drops every ChemSpec row and yields an empty join.

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
