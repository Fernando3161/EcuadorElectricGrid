# Ecuador Power System – Scenario Framework (2024–2050)
Version 1.0 — Nuclear, Renewable, and Mixed Pathways

## Overview
This document defines a consistent, reproducible, and fully transparent scenario matrix for Ecuador’s power system planning from 2017 to 2050, including:

- Calibration (2017)
- Current system (2024)
- Master Plan expansion (2030)
- Introduction of Small Modular Nuclear Reactors (SMRs) in sequential deployment waves
- Renewable-energy pathways
- Mixed nuclear + renewable pathways
- Demand growth sensitivities
- Hydrology and reliability stress tests

### SMR Deployment Roadmap
| Deployment Wave | Approx. Year | New Nuclear Capacity | Cumulative Capacity |
|-----------------|--------------|-----------------------|---------------------|
| Wave 1 | 2035 | 0.9 GW | 0.9 GW |
| Wave 2 | 2045 | +1.2 GW | 2.1 GW |
| Wave 3 | 2050 | +0.9 GW | 3.0 GW |

---

# 1. Calibration & Current System (2017–2024)

| Scenario ID | Year | Demand Case | Hydrology | Generation | Network | Purpose |
|-------------|------|-------------|-----------|------------|---------|---------|
| CAL_2017 | 2017 | Historical | 2017-like | Actual 2017 system | 2017 grid | Calibration against historical production and ENS. |
| REF_2024 | 2024 | Medium | Normal | All plants in service by 2024 | 2024 grid | Reference current system. |
| CRISIS_2024 | 2024 | Medium | Severe Dry | Same as REF_2024 | 2024 grid | Stress test representing hydro crisis. |

---

# 2. Master Plan 2030 Baseline (No Nuclear Yet)

| Scenario ID | Year | Demand Case | Hydrology | Generation | Network | Purpose |
|-------------|------|-------------|-----------|------------|---------|---------|
| REF_2030_MP | 2030 | Medium | Normal | Full Master Plan 2030 | MP grid | Future baseline. |
| RE_2030_L | 2030 | Medium | Normal | REF_2030_MP + 5% RE | MP grid | Low RE sensitivity. |
| RE_2030_H | 2030 | Medium | Normal | REF_2030_MP + 10–15% RE | MP grid | High RE sensitivity. |

---

# 3. First SMR Wave in 2035 (0.9 GW)

| Scenario ID | Year | Demand | Hydrology | Generation | Network | Purpose |
|-------------|------|--------|-----------|------------|---------|---------|
| REF_2035_MP | 2035 | Medium | Normal | REF_2030_MP + 2035 additions | MP grid 2035 | Baseline 2035. |
| NUC_2035_W1 | 2035 | Medium | Normal | +0.9 GW SMR | MP grid 2035 | Nuclear Wave 1. |
| NUC_2035_W1_H | 2035 | High | Normal | +0.9 GW SMR | MP grid 2035 | High-demand test. |
| RE_2035_H | 2035 | Medium | Normal | +10–20% RE | MP grid 2035 | RE-only alternative. |
| MIX_2035_W1_RE | 2035 | Medium | Normal | 0.9 GW SMR + moderate RE | MP grid 2035 | Hybrid path. |

---

# 4. Second SMR Wave in 2045 (2.1 GW Total)

| Scenario ID | Year | Demand | Hydrology | Generation | Network | Purpose |
|-------------|------|--------|-----------|------------|---------|---------|
| REF_2045_MP | 2045 | Medium | Normal | REF_2035 update + MP 2040 pipeline | Grid 2045 | Baseline 2045. |
| NUC_2045_W2 | 2045 | Medium | Normal | 2.1 GW nuclear | Grid 2045 | Nuclear Wave 2. |
| NUC_2045_W2_H | 2045 | High | Normal | 2.1 GW nuclear | Grid 2045 | High demand adequacy. |
| RE_2045_H | 2045 | Medium | Normal | 15–25% RE | Future grid | RE-only path. |
| MIX_2045_W2_RE | 2045 | High | Normal | Nuclear + RE | Future grid | Hybrid path. |

---

# 5. Full SMR Deployment by 2050 (3.0 GW)

| Scenario ID | Year | Demand | Hydrology | Generation | Network | Purpose |
|-------------|------|--------|-----------|------------|---------|---------|
| REF_2050_NO_NUC | 2050 | Medium | Normal | No nuclear | Grid 2050 | Alternative non-nuclear future. |
| NUC_2050_W3 | 2050 | Medium | Normal | 3.0 GW nuclear | Grid 2050 | Full nuclear scenario. |
| NUC_2050_W3_H | 2050 | High | Normal | 3.0 GW nuclear | Grid 2050 | High-demand nuclear. |
| RE_2050_XL | 2050 | Medium/High | Normal | 30–40% RE | Future grid | Very high RE scenario. |
| MIX_2050_DEEP | 2050 | High | Normal | Nuclear + strong RE | Future grid | Net-zero hybrid. |

---

# 6. Stress-Test Suffixes

| Suffix | Meaning |
|--------|---------|
| _DRY | Severe drought |
| _OUT | Major outage (plant/line) |
| _CR | Compound crisis |
| _FUEL | Fuel price shock |

Example: NUC_2035_W1_H_DRY

---

# 7. Scenario Hierarchy

- Baselines: REF_2024, REF_2030_MP, REF_2035_MP, REF_2045_MP, REF_2050_NO_NUC
- Nuclear Path: NUC_2035_W1, NUC_2045_W2, NUC_2050_W3
- RE Path: RE_2030_L, RE_2030_H, RE_2035_H, RE_2045_H, RE_2050_XL
- Mixed Path: MIX_2035_W1_RE, MIX_2045_W2_RE, MIX_2050_DEEP
- Stress tests: add suffixes (_DRY, _OUT, _CR, _FUEL)

