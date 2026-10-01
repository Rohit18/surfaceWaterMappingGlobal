| | 3 m inference | 10 m inference |
|---|---:|---:|
| Scene groups gain > 0.05 / abs(delta) <= 0.05 / loss > 0.05 | 44 / 7 / 2 | 24 / 26 / 3 |
| Mean per-scene IoU S1-only / S1+AEF(t) (pair-valid pixels) | 0.4440 / 0.7558 | 0.6112 / 0.7390 |
| **corrected false negatives**, pixels | 4,768,895 | 1,381,513 |
| &nbsp;&nbsp;Permanent water | 89.6% (4,273,458) | 77.3% (1,067,274) |
| &nbsp;&nbsp;Mangroves | 4.8% (230,710) | 13.2% (182,789) |
| &nbsp;&nbsp;Bare / sparse vegetation | 1.8% (84,825) | 3.8% (52,715) |
| &nbsp;&nbsp;Tree cover | 1.5% (71,460) | 2.2% (30,618) |
| &nbsp;&nbsp;Grassland | 0.9% (40,985) | 1.3% (17,579) |
| **corrected false positives**, pixels | 404,341 | 843,583 |
| &nbsp;&nbsp;Bare / sparse vegetation | 41.5% (167,739) | 59.3% (500,358) |
| &nbsp;&nbsp;Grassland | 36.2% (146,210) | 9.9% (83,665) |
| &nbsp;&nbsp;Tree cover | 14.1% (56,911) | 20.1% (169,470) |
| &nbsp;&nbsp;Cropland | 2.8% (11,344) | 1.2% (10,154) |
| &nbsp;&nbsp;Shrubland | 2.3% (9,129) | 2.8% (23,887) |
| &nbsp;&nbsp;Permanent water | 2.1% (8,597) | 4.1% (34,314) |
| **added false negatives**, pixels | 71,487 | 132,460 |
| &nbsp;&nbsp;Bare / sparse vegetation | 58.6% (41,884) | 48.6% (64,357) |
| &nbsp;&nbsp;Grassland | 21.8% (15,614) | 22.0% (29,205) |
| &nbsp;&nbsp;Permanent water | 11.1% (7,935) | 17.3% (22,877) |
| &nbsp;&nbsp;Shrubland | 4.5% (3,208) | 3.9% (5,231) |
| &nbsp;&nbsp;Tree cover | 2.5% (1,783) | 5.2% (6,902) |
| **added false positives**, pixels | 1,345,461 | 814,357 |
| &nbsp;&nbsp;Permanent water | 62.6% (842,876) | 61.7% (502,584) |
| &nbsp;&nbsp;Mangroves | 12.4% (166,667) | 13.5% (109,635) |
| &nbsp;&nbsp;Bare / sparse vegetation | 7.6% (101,798) | 7.3% (59,737) |
| &nbsp;&nbsp;Tree cover | 6.2% (83,135) | 7.8% (63,590) |
| &nbsp;&nbsp;Grassland | 3.5% (47,691) | 2.8% (22,404) |
