# Literature provenance catalog

Status: working catalog, links checked 2026-10-27 (Europe/Berlin), before Phase 10 pruning.

This catalog records the material currently present under `literature/`. It is an
evidence register, not a claim that every item is authoritative, redistributable,
or suitable as a model input. SHA-256 values identify the files in this checkout.
The timed-out SERCOP item and legacy files are intentionally retained until their
provenance is migrated or the owner approves their disposition.

## Distribution and Zenodo decision

Do not upload the complete `literature/` directory to Zenodo yet. A Google Drive
copy or link establishes availability, not copyright or redistribution permission.
Before publication, record for each primary document its issuing organization,
original URL, edition/date, copyright or reuse terms, and redistribution status.
For news articles and procurement attachments, preserve a citation, URL, access
date, and factual extract or metadata unless redistribution rights are clear. The
timed-out attachment should remain a URL record, not an unverifiable download.

The safer release pattern is a Zenodo deposit containing this catalog, small
normalized tables, checksums, scripts, and only documents that are clearly
redistributable. For uncertain PDFs, publish a manifest entry with the original
URL and checksum and instruct users to obtain the file themselves. Use a project
license for project-authored material; do not apply it to third-party documents.
A DOI identifies a deposit, but does not grant rights to its contents.

## Primary and contextual source register

| ID | Repository file | Role / checked source link | Rights status |
|---|---|---|---|
| LIT-ARCONEL-2023 | `ARCONEL/panorama_electrico_2023_ed14.pdf` | ARCONEL [Panorama Eléctrico 2023](https://arconel.gob.ec/panorama-electrico-2023/), listing edition 14 (January 2024) | verify before redistribution |
| LIT-TRANSELECTRIC-2022 | `CELEC_TRANSELECTRIC/mapa_snt_diciembre_2022.pdf` | CELEC-TRANSELECTRIC [PME page](https://www.celec.gob.ec/transelectric/plan-maestro-de-electricidad-2012-2022/); exact map PDF URL not exposed by checked page | verify before redistribution |
| LIT-CENACE-2022 | `CENACE/cenace_informe_anual_2022_parte_1.pdf` | [Official CENACE PDF](https://www.cenace.gob.ec/wp-content/uploads/downloads/2023/04/Parte-1-Informe-Anual-2022.pdf) | verify official terms |
| LIT-CENACE-2023 | `CENACE/cenace_informe_anual_2023_parte_1.pdf` | [Official CENACE PDF](https://www.cenace.gob.ec/wp-content/uploads/downloads/2024/04/Parte-1-Informe-Anual-CENACE-2023.pdf) | verify official terms |
| LIT-PME-ADJ-2026 | `PlanMaestroElectrifEc/AJUSTE-PME-2026.pdf` | [CENACE PME library](https://www.cenace.gob.ec/biblioteca/) lists “Ajustes al plan maestro 2023-2032 2”; [ELECAUSTRO PME page](https://www.elecaustro.gob.ec/plan-maestro-de-electricidad/) also lists current PME material | verify before redistribution |
| LIT-PME-DEMAND | `PlanMaestroElectrifEc/cap_3_demanda.pdf` | [Google Drive file](https://drive.google.com/file/d/1GSecYLLeaNrwILE2SmSYtwB7dv6ATwLg/view); official PME context via [CENACE](https://www.cenace.gob.ec/plan-maestro-de-electricidad/) | Drive access is not redistribution permission |
| LIT-PME-GEN | `PlanMaestroElectrifEc/cap_4_generacion.pdf` | [Google Drive file](https://drive.google.com/file/d/1bHNtzVuJF_GjXhh2RompVxjmwhRHcc-J/view); official PME context via [CELEC](https://www.celec.gob.ec/transelectric/plan-maestro-de-electricidad-2012-2022/) | Drive access is not redistribution permission |
| LIT-PME-TRANS | `PlanMaestroElectrifEc/cap_5_transmision.pdf` | [Google Drive file](https://drive.google.com/file/d/1Nzo6BMr-Hho9MdZvlBSa9piOH1MhWVjZ/view?usp=sharing); official transmission-plan context via [CELEC](https://www.celec.gob.ec/transelectric/plan-maestro-de-electricidad-2012-2022/) | Drive access is not redistribution permission |

The checked official landing pages confirm the publication families, but do not
prove that every local binary is byte-identical to the current online download.
The local SHA-256 values below identify the checked-in evidence.

## Drought / adequacy evidence

| File | Description / status |
|---|---|
| `draught/MEM-MEM-2024-1154-OF.pdf` | Ministry memorandum hosted by ELECAUSTRO; downloaded; verify redistribution terms |
| `draught/CELEC_SERCOP_terms_of_reference.url.txt` | [Original SERCOP attachment URL](https://www.compraspublicas.gob.ec/ProcesoContratacion/compras/GE/ExeGENBajarArchivoGeneral.cpe?Archivo=JnKh-gaCQ9zNXA4puqXl1MrgJcu3kgUy953HvOm4ZI4%2C&idPath=FfmWP6fB-wnd20gVTYELl8QfTSN_uwytR0GejxCFNHM%2C); timed out after 20 seconds on 2026-10-26; URL record only |
| `draught/planv_grinch_apagones.html` | [Plan V article](https://planv.com.ec/ideas/ideas-ideas/grinch-apagones/) snapshot; cite/archive cautiously |
| `draught/primicias_866mw.html` | [Primicias article](https://www.primicias.ec/economia/cortes-luz-lluvia-cortes-luz-futuro-deficit-colombia-electricidad108902-108902/) snapshot; cite/archive cautiously |

## Fingerprints of requested primary files

| File | Bytes | SHA-256 |
|---|---:|---|
| `ARCONEL/panorama_electrico_2023_ed14.pdf` | 9396528 | `0379cce6f2213c39cc593ff8a51dac8134a0157d5ec62c10af7be059a194fdf7` |
| `CELEC_TRANSELECTRIC/mapa_snt_diciembre_2022.pdf` | 7842142 | `755e8811f37686e2eee39373d3069d06217c9c58bf7fc8da9ef6779764e8603b` |
| `CENACE/cenace_informe_anual_2022_parte_1.pdf` | 1985889 | `88f13f0106a190464c8954babc7529f785783d4d4017633a8d5ed8ff554fbd88` |
| `CENACE/cenace_informe_anual_2023_parte_1.pdf` | 1439450 | `194a10b71b95e65c067df88756d0dbfb52052cf4713af044fec952d662f1153f` |
| `PlanMaestroElectrifEc/AJUSTE-PME-2026.pdf` | 359628 | `4afeb548ec5f7c15670ba2cb4b19c1381a6c3faf1fd3494e9b540c744fba80b0` |
| `PlanMaestroElectrifEc/cap_3_demanda.pdf` | 12643931 | `4a373f2058143d15dc1c1717f7869cefebb5b198dd16f33b06905369aee5c816` |
| `PlanMaestroElectrifEc/cap_4_generacion.pdf` | 45250553 | `009b7e855d92a67aa90ef53d8255b7eb3cfe8b49a07e5dd2a028e189b93cbf7b` |
| `PlanMaestroElectrifEc/cap_5_transmision.pdf` | 9539278 | `7bdb3bb8bfdd66173935f8952477e8f17f2a6c0dc32267c72f574b57a2b91ceb` |
| `draught/MEM-MEM-2024-1154-OF.pdf` | 138730 | `aeb23cbc364cf8169cc0a1b32776ffc1fcd54ef94e91f5e3f81de5381d5701fe` |
| `draught/CELEC_SERCOP_terms_of_reference.url.txt` | 403 | `13e89361f748d77d8c655e5c8e5681109cb32fe63c55214e5c752701c326c37d` |

Links above were checked on 2026-10-27. A checked landing page or article URL
does not establish permission to redistribute its content. The complete checksum
list, including every extract and note, can be regenerated
with `Get-ChildItem literature -Recurse -File | Get-FileHash -Algorithm SHA256`.
