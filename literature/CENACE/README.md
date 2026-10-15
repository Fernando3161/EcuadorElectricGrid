# CENACE annual reports used by the 2022 validation

These files are committed primary-source evidence for the Phase 3 statistical
contract. The normalized CSV extracts retain page and table locators; notebooks
do not parse these PDFs at runtime.

| Repository file | Official publication | Bytes | SHA-256 | Used for |
|---|---|---:|---|---|
| `cenace_informe_anual_2022_parte_1.pdf` | [Informe Anual 2022 - Parte 1](https://www.cenace.gob.ec/wp-content/uploads/downloads/2023/04/Parte-1-Informe-Anual-2022.pdf) | 1,985,889 | `88f13f0106a190464c8954babc7529f785783d4d4017633a8d5ed8ff554fbd88` | Annual SNI demand and peak; net supply by generation type and imports, especially Table 2.4 on pages 73-74 |
| `cenace_informe_anual_2023_parte_1.pdf` | [Informe Anual 2023 - Parte 1](https://www.cenace.gob.ec/wp-content/uploads/downloads/2024/04/Parte-1-Informe-Anual-CENACE-2023.pdf) | 1,439,450 | `194a10b71b95e65c067df88756d0dbfb52052cf4713af044fec952d662f1153f` | Table 1.8 on page 42, which republishes monthly 2022 SNI demand and peaks |

The files were downloaded from the official CENACE URLs on 2026-10-15. Both
passed a binary integrity check for a PDF header, cross-reference trailer, and
end-of-file marker. No PDF parsing or rendering dependency was installed during
the download; the cited tables had already been transcribed and independently
checked during Phase 3.
