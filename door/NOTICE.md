# DoOR odour data

The files in this folder are unmodified copies from **DoOR.data**, the data
package of the Database of Odorant Responses:

- source: https://github.com/Dahaniel/DoOR.data (version 2.0.1.9001, see `DESCRIPTION`)
- files: `door_response_matrix.csv`, `door_mappings.csv`, `odor.csv`,
  `door_dataset_info.csv`, `DESCRIPTION`

**Licence.** These data files are licensed under
[Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
They are not covered by this repository's MIT licence. If you redistribute or
adapt them, the CC BY-SA 4.0 terms apply.

**Cite.** Münch D, Galizia CG (2016). DoOR 2.0 – Comprehensive Mapping of
Drosophila melanogaster Odorant Responses. *Scientific Reports* 6, 21841.
https://doi.org/10.1038/srep21841

**How they are used here.** `olfaction.py` reads the receptor response matrix
and the receptor-to-glomerulus mapping, and drives this connectome's
`ORN_<glomerulus>` neurons at response × 200 Hz. Which odorant a coin smells of
is a chosen convention in `olfaction.py`, not part of DoOR.

Note for readers of the CSVs: they were written by R, so data rows in
`door_mappings.csv` carry a leading row-name column that has no header cell.
