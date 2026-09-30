# Figure 1B scene selection (10 m inference)

Values: 10 m inference, probabilities resampled bilinearly to the 3 m label grid, threshold 0.30, IoU on the paper's common valid mask; gain = IoU(S1+AEF(t)) - IoU(S1-only). Source: `fig1B/scene_table.csv` (written by `scripts/export_53_scene_layers_10m.py`). Continent: Natural Earth 1:50m admin-0 `CONTINENT` at the scene centre from `S1_Intercomparison_All_Scenes.csv` (point in polygon; SID15, SID46 and SID55 lie in no polygon and take the nearest one).

## Rule

1. Eligible scenes: label water fraction in [0.05, 0.60], common-valid fraction >= 0.95, and not one of the current Fig. 1B scenes (SID23, SID46, SID17).
2. Role C, large AEF gain: gain > 0.20; ranked by gain, largest first.
3. Role B, near the median gain: the median gain over all 53 scenes is +0.0337; ranked by |gain - median|, smallest first.
4. Role A, S1-only already good and AEF changes little: S1-only IoU >= 0.80 and |gain| <= 0.05; ranked by |gain|, smallest first.
5. A set is filled in the order C, B, A. Each role takes its best-ranked candidate that is not already in the set and whose continent differs from those already in the set. Ties are broken by SID number.
6. alt1 is the first set. alt2 is built by the same procedure after also excluding the alt1 scenes.
7. Rows are drawn in the order C, B, A. Row labels: SID and country (United States of America shortened to USA).

This is the suggested rule of the brief, with two additions needed to make it deterministic: the |gain| <= 0.05 bound for "AEF changes little" (the same band as the WorldCover no-change group) and the fixed filling order and ranking above.

Eligible scenes: 26 of 53.

### Candidates for role C (large gain (> 0.20)), in rank order

| Scene | Country | Continent | Label water fraction | Common-valid fraction | S1-only | S1+AEF(t) | Gain | rank: gain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SID33 | Indonesia | Asia | 0.2695 | 0.9998 | 0.1570 | 0.6430 | +0.4860 | 0.4860 |
| SID83 | Thailand | Asia | 0.1475 | 1.0000 | 0.5331 | 0.9444 | +0.4113 | 0.4113 |
| SID98 | Kazakhstan | Asia | 0.0674 | 1.0000 | 0.2070 | 0.4619 | +0.2549 | 0.2549 |
| SID97 | Uzbekistan | Asia | 0.1982 | 1.0000 | 0.4199 | 0.6744 | +0.2544 | 0.2544 |

### Candidates for role B (near the median gain), in rank order

| Scene | Country | Continent | Label water fraction | Common-valid fraction | S1-only | S1+AEF(t) | Gain | rank: abs(gain - median) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SID39 | Canada | North America | 0.3091 | 1.0000 | 0.9041 | 0.9378 | +0.0337 | 0.0000 |
| SID03 | United States of America | North America | 0.3404 | 1.0000 | 0.8647 | 0.9036 | +0.0389 | 0.0052 |
| SID52 | Spain | Europe | 0.1462 | 1.0000 | 0.8467 | 0.8735 | +0.0268 | 0.0069 |
| SID05 | Finland | Europe | 0.5310 | 1.0000 | 0.9117 | 0.9373 | +0.0256 | 0.0081 |
| SID53 | Pakistan | Asia | 0.4485 | 1.0000 | 0.7934 | 0.8357 | +0.0424 | 0.0087 |
| SID21 | Peru | South America | 0.2896 | 0.9789 | 0.8962 | 0.9114 | +0.0152 | 0.0185 |
| SID80 | Malaysia | Asia | 0.0874 | 1.0000 | 0.8681 | 0.8826 | +0.0145 | 0.0192 |
| SID14 | Brazil | South America | 0.1215 | 1.0000 | 0.8328 | 0.8465 | +0.0137 | 0.0200 |
| SID69 | Canada | North America | 0.1150 | 1.0000 | 0.6815 | 0.7369 | +0.0554 | 0.0217 |
| SID81 | Brazil | South America | 0.2827 | 1.0000 | 0.9692 | 0.9808 | +0.0115 | 0.0222 |
| SID64 | Russia | Europe | 0.0616 | 1.0000 | 0.7807 | 0.8375 | +0.0568 | 0.0231 |
| SID40 | France | Europe | 0.5839 | 1.0000 | 0.9719 | 0.9815 | +0.0096 | 0.0241 |
| SID36 | Russia | Europe | 0.1543 | 1.0000 | 0.7339 | 0.7934 | +0.0595 | 0.0258 |
| SID78 | United States of America | North America | 0.2562 | 1.0000 | 0.9257 | 0.9274 | +0.0017 | 0.0320 |
| SID63 | Finland | Europe | 0.4858 | 0.9999 | 0.8655 | 0.9314 | +0.0659 | 0.0322 |
| SID04 | Vietnam | Asia | 0.1559 | 1.0000 | 0.7713 | 0.7712 | -0.0001 | 0.0338 |
| SID85 | Mexico | North America | 0.3576 | 0.9834 | 0.9825 | 0.9807 | -0.0018 | 0.0355 |
| SID59 | United States of America | North America | 0.0839 | 1.0000 | 0.7939 | 0.7845 | -0.0094 | 0.0431 |
| SID61 | Finland | Europe | 0.1842 | 1.0000 | 0.8247 | 0.9135 | +0.0887 | 0.0550 |
| SID55 | Pakistan | Asia | 0.4596 | 1.0000 | 0.5900 | 0.5635 | -0.0264 | 0.0601 |
| SID18 | Australia | Oceania | 0.0518 | 1.0000 | 0.8768 | 0.8204 | -0.0563 | 0.0900 |
| SID97 | Uzbekistan | Asia | 0.1982 | 1.0000 | 0.4199 | 0.6744 | +0.2544 | 0.2207 |
| SID98 | Kazakhstan | Asia | 0.0674 | 1.0000 | 0.2070 | 0.4619 | +0.2549 | 0.2212 |
| SID28 | Pakistan | Asia | 0.5003 | 0.9993 | 0.9684 | 0.7712 | -0.1972 | 0.2309 |
| SID83 | Thailand | Asia | 0.1475 | 1.0000 | 0.5331 | 0.9444 | +0.4113 | 0.3776 |
| SID33 | Indonesia | Asia | 0.2695 | 0.9998 | 0.1570 | 0.6430 | +0.4860 | 0.4523 |

### Candidates for role A (S1-only >= 0.80, |gain| <= 0.05), in rank order

| Scene | Country | Continent | Label water fraction | Common-valid fraction | S1-only | S1+AEF(t) | Gain | rank: abs(gain) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SID78 | United States of America | North America | 0.2562 | 1.0000 | 0.9257 | 0.9274 | +0.0017 | 0.0017 |
| SID85 | Mexico | North America | 0.3576 | 0.9834 | 0.9825 | 0.9807 | -0.0018 | 0.0018 |
| SID40 | France | Europe | 0.5839 | 1.0000 | 0.9719 | 0.9815 | +0.0096 | 0.0096 |
| SID81 | Brazil | South America | 0.2827 | 1.0000 | 0.9692 | 0.9808 | +0.0115 | 0.0115 |
| SID14 | Brazil | South America | 0.1215 | 1.0000 | 0.8328 | 0.8465 | +0.0137 | 0.0137 |
| SID80 | Malaysia | Asia | 0.0874 | 1.0000 | 0.8681 | 0.8826 | +0.0145 | 0.0145 |
| SID21 | Peru | South America | 0.2896 | 0.9789 | 0.8962 | 0.9114 | +0.0152 | 0.0152 |
| SID05 | Finland | Europe | 0.5310 | 1.0000 | 0.9117 | 0.9373 | +0.0256 | 0.0256 |
| SID52 | Spain | Europe | 0.1462 | 1.0000 | 0.8467 | 0.8735 | +0.0268 | 0.0268 |
| SID39 | Canada | North America | 0.3091 | 1.0000 | 0.9041 | 0.9378 | +0.0337 | 0.0337 |
| SID03 | United States of America | North America | 0.3404 | 1.0000 | 0.8647 | 0.9036 | +0.0389 | 0.0389 |

Scenes excluded by the eligibility filters: SID06 (water fraction 0.7158), SID15 (water fraction 0.7518), SID17 (water fraction 0.0356), SID22 (common-valid 0.8131), SID25 (common-valid 0.2823), SID26 (water fraction 0.0195), SID27 (water fraction 0.0094; common-valid 0.7481), SID31 (water fraction 0.0325), SID37 (water fraction 0.8606), SID44 (common-valid 0.8652), SID46 (water fraction 0.6007), SID50 (water fraction 0.0089; common-valid 0.5726), SID51 (water fraction 0.0308), SID65 (common-valid 0.6855), SID67 (water fraction 0.0248), SID68 (water fraction 0.0396; common-valid 0.9404), SID72 (water fraction 0.8837), SID79 (common-valid 0.7295), SID87 (water fraction 0.0216), SID89 (common-valid 0.6712), SID90 (water fraction 0.0381; common-valid 0.6264), SID91 (water fraction 0.0356), SID92 (common-valid 0.8728), SID93 (water fraction 0.0299), SID99 (water fraction 0.0054), SID100 (water fraction 0.0396; common-valid 0.8390).

## Chosen sets and panel IoUs

### current

| Row | Scene | Role | Country / continent | S1-only | S1+AEF(t) | OPERA | Gain | 3 m inference: S1-only / S1+AEF(t) |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | SID23 | current Fig. 1B | Romania / Europe | 0.0233 | 0.8695 | 0.8508 | +0.8462 | 0.0059 / 0.9019 |
| 2 | SID46 | current Fig. 1B | Vietnam / Asia | 0.3744 | 0.8008 | 0.4751 | +0.4264 | 0.1099 / 0.8275 |
| 3 | SID17 | current Fig. 1B | Spain / Europe | 0.6417 | 0.6057 | 0.2738 | -0.0360 | 0.3179 / 0.6674 |

S1-only beats S1+AEF(t): SID17 (0.64 vs 0.61).

### alt1

| Row | Scene | Role | Country / continent | S1-only | S1+AEF(t) | OPERA | Gain | 3 m inference: S1-only / S1+AEF(t) |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | SID33 | large gain (> 0.20) | Indonesia / Asia | 0.1570 | 0.6430 | 0.6120 | +0.4860 | 0.0548 / 0.7023 |
| 2 | SID39 | near the median gain | Canada / North America | 0.9041 | 0.9378 | 0.8840 | +0.0337 | 0.7183 / 0.9481 |
| 3 | SID40 | S1-only >= 0.80, |gain| <= 0.05 | France / Europe | 0.9719 | 0.9815 | 0.9235 | +0.0096 | 0.9001 / 0.9892 |

S1-only beats S1+AEF(t): none.

### alt2

| Row | Scene | Role | Country / continent | S1-only | S1+AEF(t) | OPERA | Gain | 3 m inference: S1-only / S1+AEF(t) |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | SID83 | large gain (> 0.20) | Thailand / Asia | 0.5331 | 0.9444 | 0.9189 | +0.4113 | 0.0438 / 0.9517 |
| 2 | SID03 | near the median gain | United States of America / North America | 0.8647 | 0.9036 | 0.7793 | +0.0389 | 0.8202 / 0.9217 |
| 3 | SID81 | S1-only >= 0.80, |gain| <= 0.05 | Brazil / South America | 0.9692 | 0.9808 | 0.9210 | +0.0115 | 0.9377 / 0.9892 |

S1-only beats S1+AEF(t): none.

