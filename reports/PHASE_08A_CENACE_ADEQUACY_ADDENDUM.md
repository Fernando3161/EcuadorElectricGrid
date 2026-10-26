# Addendum to Phase 08A — External context from CENACE adequacy assessments

**Related report:** `PHASE_08A_EL_NINO_DROUGHT_REPORT.md`  
**Prepared:** 26 October 2026

## Purpose

This addendum places the Phase 08A drought-screening result in the context of Ecuadorian generation-adequacy assessments. It asks whether a transparent model built from open data points to a real system-planning concern at a broadly comparable scale. It does not attempt a number-for-number validation: the model and CENACE use different base years, assumptions, operating conditions, and adequacy criteria.

## What the official planning record says

In an official letter dated 10 September 2024, the Ministry of Energy and Mines (MEM) relayed CENACE's assessment that **1,080 MW of additional firm generation** was needed for the 2024–2025 dry-season period: 500 MW for April–August 2024 and another 580 MW for September 2024–March 2025. The assessment aimed to maintain a **10% energy reserve without relying on international interconnections**. The letter reports that 341 MW had entered the system and that **749 MW remained** to meet the 1,080 MW requirement. The signed government letter is hosted by the public generator ELECAUSTRO, which received the Ministry's instruction. [MEM Oficio Nro. MEM-MEM-2024-1154-OF](https://www.elecaustro.gob.ec/download/oficio-nro-mem-mem-2024-1154-of/?refresh=69cbd142b5e6a1774965058&wpdmdl=15875).

For the next dry-season cycle, CENACE's **14 August 2025** assessment raised the total generation requirement to **1,510 MW**: the previous 1,080 MW plus a further 430 MW for 2025–2026. The letter, **CENACE-CENACE-2025-1187-O**, reported that 590 MW had entered by 13 August 2025, with a further 33.5 MW from ELECAUSTRO and 20 MW at Quevedo reported as expected. On that accounting, **866.5 MW remained**. The CENACE letter is identified as the technical basis in CELEC procurement documents published through Ecuador's public procurement portal; its figures are quoted from the letter in contemporaneous reporting. [CELEC terms of reference hosted by SERCOP](https://www.compraspublicas.gob.ec/ProcesoContratacion/compras/GE/ExeGENBajarArchivoGeneral.cpe?Archivo=JnKh-gaCQ9zNXA4puqXl1MrgJcu3kgUy953HvOm4ZI4%2C&idPath=FfmWP6fB-wnd20gVTYELl8QfTSN_uwytR0GejxCFNHM%2C); [Plan V, 18 December 2025](https://planv.com.ec/ideas/ideas-ideas/grinch-apagones/); [Primicias, 8 November 2025](https://www.primicias.ec/economia/cortes-luz-lluvia-cortes-luz-futuro-deficit-colombia-electricidad108902-108902/).

The distinctions matter: **1,510 MW is the total planned addition**, whereas **866.5 MW is the remaining requirement after additions counted by CENACE at that date**. These are planning quantities, not direct observations of load shedding. The cited reports also state that the applicable Plan Maestro de Electricidad criteria include a 10% energy reserve for dry scenarios and a 20% power reserve. Those reserve margins make CENACE's planning need broader than the capacity required merely to serve one modeled peak hour.

## What Phase 08A measures

Phase 08A uses a **2022 installed-generation fleet** and a 2026 demand proxy based on the 2025 annual-demand target. It applies a 50% reduction to four selected river-flow series during September–December in case H50. In that case, the report records 328.8 GWh of annual load shedding and a December peak hourly shedding value of 1,137.9 MW.

The report's approximately **1,638 MW** indicator is calculated as:

```text
P_independence = max_t(Colombia imports_t + load shedding_t) ≈ 1,638 MW
```

This is the maximum coincident hourly sum in H50. It represents the modeled domestic firm-supply equivalent needed at that critical hour to replace Colombian imports and serve the demand still unserved by the model. It is a **scenario-specific independence benchmark**, not a measured national capacity deficit and not, by itself, a nameplate-build target. It contains no added reserve margin, and the installed capacity needed in practice could differ due to outages, unit availability, plant location, ramping, and network constraints. The Phase 08A network also uses a 500 MW Colombia import limit; its report notes that a separately documented 450 MW transfer figure was not imposed in these solves.

## Interpretation

The comparison supports a careful, positive conclusion. CENACE's later planning assessments identify a need for additional firm generation on the order of **1 GW or more** to maintain adequate supply and reserves through dry periods, including without dependence on imports. Independently, Phase 08A's severe-hydrology case produces an approximately **1.64 GW coincident domestic-supply requirement** if the modeled system is to replace imports and eliminate its residual peak shedding. The starting fleets differ: the model represents 2022 generation, while CENACE's 2025 accounting includes post-2022 additions. Therefore, the similarity in broad scale is contextual consistency, not a direct equivalence.

Taken together, these results indicate that Phase 08A is reflecting a **real system-adequacy problem identified by the national system operator**, rather than producing an implausible concern unique to this model. The fact that an open-data, transparent screening workflow reaches the same broad conclusion—substantial additional firm supply is needed to withstand dry conditions and reduce reliance on Colombian imports—supports using this modelling approach for exploratory analysis and further development.

This does **not** establish that H50 is a forecast, that a 1.64 GW installation is the correct policy response, or that the model is quantitatively validated. The drought severity is a scenario assumption; hydrological profiles are simplified proxies; the demand shape and fleet year differ from CENACE's current operational/planning inputs. The result is best presented as external plausibility evidence and a reason to improve the model, not as a forecast or a verified capacity-procurement target.

## Implications for further model development

The comparison suggests a credible next phase: retain the open and reproducible workflow, then improve the inputs and align selected cases with official planning conditions. Priorities are to reconcile the fleet with post-2022 additions and retirement/availability status; test CENACE-aligned reserve and import assumptions; replace the selected-river proxy shocks with observed basin inflow anomalies and explicit reservoir states; and document how the peak independence indicator changes under those sensitivities. Such steps would move the model from exploratory screening toward a more defensible adequacy study.

## Sources and evidentiary notes

1. **Primary government document:** Ministry of Energy and Mines, Oficio Nro. MEM-MEM-2024-1154-OF, 10 September 2024. The official memorandum reproduces the 2024–2025 CENACE requirement and the remaining need after additions: [ELECAUSTRO-hosted official copy](https://www.elecaustro.gob.ec/download/oficio-nro-mem-mem-2024-1154-of/?refresh=69cbd142b5e6a1774965058&wpdmdl=15875).
2. **CENACE 2025 assessment and official procurement trail:** CENACE, Oficio Nro. CENACE-CENACE-2025-1187-O, 14 August 2025. The letter is identified as an input to CELEC's generation procurement in this [terms-of-reference document hosted by SERCOP](https://www.compraspublicas.gob.ec/ProcesoContratacion/compras/GE/ExeGENBajarArchivoGeneral.cpe?Archivo=JnKh-gaCQ9zNXA4puqXl1MrgJcu3kgUy953HvOm4ZI4%2C&idPath=FfmWP6fB-wnd20gVTYELl8QfTSN_uwytR0GejxCFNHM%2C). The quoted requirement and additions are reported by [Plan V](https://planv.com.ec/ideas/ideas-ideas/grinch-apagones/) and [Primicias](https://www.primicias.ec/economia/cortes-luz-lluvia-cortes-luz-futuro-deficit-colombia-electricidad108902-108902/), both of which identify the CENACE letter as their source.
3. **Plan Maestro reserve criteria:** the 10% energy and 20% power reserve criteria are described in Primicias' account of CENACE's assessment above. The official MEM Plan Maestro chapters supplied with this project also set out dry and extra-dry hydrological planning cases and the determination of power and energy reserves in the generation-expansion methodology.
4. **Model results and assumptions:** Phase 08A report, `PHASE_08A_EL_NINO_DROUGHT_REPORT.md`, especially Sections 2, 3, 9, and its import-cap caveat. Its reported results are the basis for the 1,638 MW comparison and the model-description statements in this addendum.

*Prepared 26 October 2026. Source figures are reported as stated in the cited documents and reporting; no independent recalculation of CENACE's planning study was performed here.*
