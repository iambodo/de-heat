# Paper Synopsis: Heat-Mortality Germany (ClimSocAna 2024)

**Full title:** High-resolution modeling and projection of heat-related mortality in Germany under climate change  
**Journal:** Communications Medicine (Nature Portfolio), 2024  
**DOI:** https://doi.org/10.1038/s43856-024-00643-3  
**Code:** https://github.com/ClimSocAna/Heat-Mortality (branch: Publication)

---

## What the paper does

The paper builds a district-level (Kreise, n≈400) statistical model linking daily temperature to all-cause mortality across Germany, then projects future heat-attributable deaths under climate change scenarios.

It answers two questions:
1. What is the current temperature–mortality relationship at district level, and how does it vary spatially?
2. How will heat-attributable mortality change under SSP2-4.5 and SSP5-8.5 warming scenarios?

---

## Data

| Input | Source | Granularity |
|-------|--------|-------------|
| All-cause daily deaths | Destatis / state statistical offices | Daily, by Kreise, ~2000–2021 |
| Daily mean 2m temperature | ERA5 reanalysis (2000–2020) + DWD station interpolation (2021–2023) | Daily, by Kreise centroid |
| Population | Destatis | Annual, by Kreise |
| Future temperature projections | EURO-CORDEX regional climate models | Daily, bias-corrected to ERA5 |

---

## Statistical model: shallow neural network

> **Note:** The actual code uses a **PyTorch shallow neural network**, not an R DLNM. See `docs/code_analysis.md` for the full analysis. The description below reflects the actual implementation.

The model is a two-component PyTorch / PyTorch Lightning system:

### TempModel — station-to-district interpolation

An attention-based spatial interpolator that converts DWD weather station point measurements to district-level daily temperature for the 400 Kreise. Learned parameters include optimal district centroid positions and distance-decay scales.

### MortModel — mortality prediction

A shallow neural network (`HeatMortality_EXP`) that takes the entire daily temperature time series for all 400 Kreise and outputs predicted deaths per Kreise × age group × sex.

**Lag effect:** captured via a `Conv1d` kernel spanning `kernel_days=6` — the network learns which days in the preceding 6 days contribute to today's mortality, without an explicit lag specification.

**Non-linearity:** the exponential activation branches give the model capacity to represent the steep rise in mortality at extreme temperatures.

**Baseline:** instead of explicit seasonality terms, uses the rolling mean of the 10 lowest weekly death counts in the past year ("bottom 10") as the expected baseline. The model predicts a *multiplier* on this baseline.

**Supervision:** despite predicting at Kreise level, the model is trained against Bundesland and Germany totals only (Kreise-level daily death data is not publicly available). A multi-scale Poisson NLL loss supervises at both daily/Bundesland and weekly/Germany granularities simultaneously.

**Weekday correction:** 6 learnable scalars correct for systematic under-reporting of deaths on weekends.

---

## Climate projections

- **Scenarios**: SSP2-4.5 (moderate) and SSP5-8.5 (high emissions)
- **Models**: ensemble of EURO-CORDEX regional climate models, bias-corrected against ERA5
- **Periods**: near-term (2021–2040), mid-century (2041–2060), end-of-century (2081–2100)
- Applies the fitted DLNM coefficients to projected temperature distributions
- Holds the exposure–response function constant (does not model adaptation)

Key finding: heat-attributable deaths approximately double under SSP5-8.5 by end of century, with the largest increases in urban Kreise in the south and west.

---

## How to replicate this in DHIS2 / CHAP

### What we have

| Required | Available in our setup |
|----------|----------------------|
| Weekly mortality by Bundesland | ✓ Imported (2000–2026) |
| Weekly temperature by Kreise | ✓ Importing (ERA5 via Open-Meteo) |
| Population by Bundesland | Partially (need 2025–2026) |
| Kreis-level mortality | ✗ Not publicly available |

### Key differences from the paper

1. **Spatial resolution**: Paper uses Kreise (~400); we have Bundesland (16) for mortality. Temperature is at Kreise level. Model will operate at Bundesland level unless Kreise mortality is sourced.

2. **Temporal resolution**: Paper uses daily; CHAP uses weekly (our period type). The cross-basis lag structure needs to be adapted from 21 daily lags to ~3 weekly lags.

3. **Model framework**: Paper uses R (`dlnm`). CHAP expects Python models (MLflow pyfunc). Options:
   - Re-implement DLNM in Python using `statsmodels` GLM + manual cross-basis splines
   - Wrap R model in a Python subprocess (rpy2 or shell call)
   - Simplify to a quasi-Poisson GLM with polynomial temperature terms (loses lag structure)

4. **Training signal**: Heat signal is clearer at Kreise level. At Bundesland level, spatial averaging smooths out local hotspots and may reduce effect size estimates.

### Recommended CHAP adaptation

Given available data, the most scientifically defensible approach is:

**Stage 1 — Bundesland model (achievable now)**
- Quasi-Poisson GLM per Bundesland with:
  - Natural spline on weekly mean temperature (3 knots)
  - Lag structure: current week + 1-week lag (limited by weekly resolution)
  - Day-of-year seasonality (Fourier terms or spline)
  - Log(population) offset
  - Linear year trend
- Predict weekly deaths; convert to heat-attributable fraction above MMT

**Stage 2 — Kreise model (when Kreise mortality is available)**
- Full DLNM as in paper, operating at Kreise level
- Two-stage meta-analysis to pool across districts

### CHAP input/output mapping

| CHAP field | Paper equivalent |
|------------|-----------------|
| `time_period` | ISO week (YYYY-Wnn) |
| `location` | Bundesland UID → name → Kreise AGS |
| `mean_temperature` | Weekly mean of daily 2m temperature |
| `population` | Annual Destatis population |
| `disease_cases` | All-cause weekly deaths |
| model output `mean` | Predicted deaths (quasi-Poisson mean) |
| model output `low/high` | 95% CI from quasi-Poisson fit |

---

## Key concepts to understand

- **MMT (Minimum Mortality Temperature)**: varies by location — southern German cities have higher MMT than northern rural areas. It is estimated from the fitted spline, not set a priori.
- **Harvesting**: heat kills vulnerable people who were already near death; some of the excess mortality is "borrowed" from subsequent weeks. The lag structure captures this.
- **Quasi-Poisson vs Negative Binomial**: quasi-Poisson is preferred here because it does not require specifying a variance-mean relationship, just allowing for overdispersion.
- **Two-stage pooling**: critical for stable estimates in districts with few hot days per year.
