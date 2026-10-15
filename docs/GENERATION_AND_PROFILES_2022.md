# 2022 generation fleet and availability profiles

## Scope

Stages 02 and 03 prepare the existing mainland Sistema Nacional Interconectado
(SNI) generator fleet and its hydro, solar, and onshore-wind availability. They
do not assemble or solve a PyPSA network. Future projects and all hypothetical
nuclear assets are excluded.

## Existing-generation source

`data/raw/generation/generation_existing_2022.csv` is the open,
source-faithful migration of the owner's existing-generation workbook. It retains
all 300 rows, the original labels and capacities, the original workbook row, and
the document locator. The source is Chapter 4, section 4.11.1, Annex 1 of the
[Electricity Master Plan generation chapter](https://www.recursosyenergia.gob.ec/wp-content/uploads/2020/01/4.-EXPANSION-DE-LA-GENERACION.pdf).

The migration is reproducible with
`scripts/migrate_phase04_generation_sources.py`. The source workbooks remain
local evidence and are not runtime inputs. A fresh user obtains the tracked CSV
without needing Excel or a manual download.

Only the 133 rows explicitly marked SNI enter the baseline. Their capacities
reconcile to 7,489.65 MW nominal and 7,175.11 MW effective. Nominal capacity is
the model-facing `p_nom_mw`; effective capacity is retained for audit and
diagnostics. The source's three cases where effective capacity is slightly above
nominal are preserved rather than altered.

## Plant identity, carriers, and location

Each plant ID combines its immutable source row and a readable name token. This
keeps the two legitimate `Loreto` rows distinct without changing their display
name. Plant-level carriers are:

| Source technology | Model carrier |
|---|---|
| Hidraulica | `hydro` |
| Termica | `oil` |
| Biomasa | `biomass` |
| Biogas | `biogas` |
| Solar | `solar` |
| Eolica | `onwind` |

The source does not distinguish thermal fuels plant by plant. Stage 06 therefore
uses the already frozen `thermal_nonrenewable` validation aggregation rather than
claiming that every thermal plant burns oil.

`data/interim/generation/plant_location_evidence_2022.csv` contains exactly one
location row per SNI source row. Its methods are exact plant/province matches to
the 2017 inventory, retained manual canton assignments, or same-province capacity
proxies within 5%. Duplicate canton lookup rows are aggregated by key only after
verifying that their coordinates differ by no more than 0.01 degrees. The
capacity proxies and manual mappings remain evidence limitations for Phase 5 bus
attachment.

The retained inventory does not provide complete row-level commissioning or
retirement dates, so Phase 4 does not infer historical dates. CENACE Table 3.1
does identify three additions during 2022: Sarapullo on 7 July, Sabanilla on 1
December, and Huascachaca unit groups on 2 and 24 December. These events are
retained in `data/raw/generation/cenace_2022_commissioning_events.csv`. Profiles
are zero before each plant's earliest official operating date. Event capacities
remain separate because they do not exactly reconcile to the Master Plan plant
capacity, especially for staged Huascachaca units.

## Hydro availability

The legacy hydrology table contains eight monthly flow series. Every one of the
65 hydro plants has a versioned assignment in
`data/interim/generation/hydro_profile_assignments_2022.csv`. The assignment
audit distinguishes:

- 4 direct plant/series matches, labeled high confidence;
- 5 same-system or same-complex mappings, labeled medium confidence; and
- 56 retained legacy regional proxies, labeled low confidence.

This is a material limitation. The proxies preserve prior manual work but are
not measured plant-specific inflows.

Run-of-river availability is the assigned monthly flow divided by its 95th
percentile and clipped to `[0, 1]`. Reservoir plants use a trailing three-month
flow mean before the same normalization. Monthly points are placed near the
middle of each month, linearly interpolated to hourly resolution, and extended
to the calendar edges. The result is a deterministic modeling constraint, not
historical 2022 dispatch or inflow.

For Sarapullo and Sabanilla, the resulting profile is zero before the official
2022 operating date. This prevents the existing-year inventory from leaking
their capacity backward into earlier months.

## Solar and wind weather year

The published `solar.nc` and `onwind.nc` inputs contain complete 2013 hourly
profiles for 24 Ecuador provinces. Stage 03 maps each SNI solar or wind plant to
its province and transfers the 8,760 hourly positions to the non-leap 2022 model
calendar. This is a transparent 2013-weather assumption for a 2022 system, not
an assertion that the weather chronology is from 2022. No full ERA5 cutout is
needed in this repository.

Huascachaca is zero before its first official unit-group date. The two staged
event capacities do not sum to the Master Plan plant capacity, so Phase 4 does
not fabricate a fractional capacity trajectory between the two dates.

## Outputs and checks

Stage 02 writes the small tracked table
`data/processed/generation/generators_2022.csv`. Stage 03 writes two large,
ignored, reproducible tables:

- `data/processed/generation/hydro_availability_2022.csv`;
- `data/processed/generation/renewable_availability_2022.csv`.

All profiles must contain exactly 8,760 timezone-naive Ecuador local-standard
hours, cover every applicable plant exactly once, contain no missing values, and
remain within `[0, 1]`. These checks do not validate annual generation; that
requires the solved-network energy balance and 5% carrier comparison in Phase 6.
