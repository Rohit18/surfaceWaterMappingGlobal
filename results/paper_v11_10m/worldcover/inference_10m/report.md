# S1-Only vs S1+AEF WorldCover Error Analysis

Created: `2026-09-29T03:35:32Z`

## Setup

- S1-only: `/pscratch/sd/r/rohit9/S1ML/revision_checks_20260930/worldcover_10m/inputs/width_k0_seed42_current_tta_10m_per_sample_inputs.csv` at threshold `0.3`
- S1+AEF: `/pscratch/sd/r/rohit9/S1ML/revision_checks_20260930/worldcover_10m/inputs/width_k16_seed42_current_tta_10m_per_sample_inputs.csv` at threshold `0.3`
- Processed chips: `53`
- ESA WorldCover cache: `/pscratch/sd/r/rohit9/S1ML/intercomparison_s1aef/worldcover/esa_worldcover_2021_v200`
- Validation groups: `permanent_water` = SID labels, `not_permanent_water` = non-SID post-flood labels

## Chip Delta IoU

- Mean S1 Water IoU: `0.6112`
- Mean S1+AEF Water IoU: `0.7390`
- Mean delta IoU: `0.1278`
- Groups: `24` helps, `26` neutral, `3` hurts

## Validation Group Summary

| Validation group | Chips | S1 IoU | S1+AEF IoU | Mean delta | Helps | Neutral | Hurts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| permanent_water | 53 | 0.6112 | 0.7390 | 0.1278 | 24 | 26 | 3 |
| not_permanent_water | 0 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 |

## WorldCover Classes

### permanent_water

- Chips: `53`
- Mean delta IoU: `0.1278`
- Groups: `24` helps, `26` neutral, `3` hurts

#### AEF fixes S1 false negatives

| Class | Pixels | Pixel fraction | Chips |
| --- | ---: | ---: | ---: |
| Permanent water | 1067274 | 0.7725 | 49 |
| Mangroves | 182789 | 0.1323 | 5 |
| Bare / sparse vegetation | 52715 | 0.0382 | 31 |
| Tree cover | 30618 | 0.0222 | 34 |
| Grassland | 17579 | 0.0127 | 33 |

#### AEF fixes S1 false positives

| Class | Pixels | Pixel fraction | Chips |
| --- | ---: | ---: | ---: |
| Bare / sparse vegetation | 500358 | 0.5931 | 27 |
| Tree cover | 169470 | 0.2009 | 36 |
| Grassland | 83665 | 0.0992 | 42 |
| Permanent water | 34314 | 0.0407 | 36 |
| Shrubland | 23887 | 0.0283 | 15 |

#### AEF adds false negatives

| Class | Pixels | Pixel fraction | Chips |
| --- | ---: | ---: | ---: |
| Bare / sparse vegetation | 64357 | 0.4859 | 13 |
| Grassland | 29205 | 0.2205 | 26 |
| Permanent water | 22877 | 0.1727 | 30 |
| Tree cover | 6902 | 0.0521 | 24 |
| Shrubland | 5231 | 0.0395 | 6 |

#### AEF adds false positives

| Class | Pixels | Pixel fraction | Chips |
| --- | ---: | ---: | ---: |
| Permanent water | 502584 | 0.6172 | 47 |
| Mangroves | 109635 | 0.1346 | 5 |
| Tree cover | 63590 | 0.0781 | 38 |
| Bare / sparse vegetation | 59737 | 0.0734 | 34 |
| Grassland | 22404 | 0.0275 | 40 |

### not_permanent_water

- Chips: `0`
- Mean delta IoU: `0.0000`
- Groups: `0` helps, `0` neutral, `0` hurts

#### AEF fixes S1 false negatives

No pixels in this category.

#### AEF fixes S1 false positives

No pixels in this category.

#### AEF adds false negatives

No pixels in this category.

#### AEF adds false positives

No pixels in this category.

## Top 10 AEF Improvements

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| SID23 | aef_helps | 0.0232 | 0.8691 | 0.8459 |
| SID100 | aef_helps | 0.1475 | 0.8208 | 0.6733 |
| SID89 | aef_helps | 0.3392 | 0.9294 | 0.5902 |
| SID33 | aef_helps | 0.1569 | 0.6429 | 0.4860 |
| SID67 | aef_helps | 0.0121 | 0.4873 | 0.4752 |
| SID27 | aef_helps | 0.0417 | 0.4738 | 0.4321 |
| SID46 | aef_helps | 0.3744 | 0.8008 | 0.4264 |
| SID83 | aef_helps | 0.5331 | 0.9444 | 0.4113 |
| SID90 | aef_helps | 0.4016 | 0.7411 | 0.3395 |
| SID26 | aef_helps | 0.0000 | 0.2683 | 0.2683 |

## Bottom 10 AEF Regressions

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| SID28 | aef_hurts | 0.9684 | 0.7712 | -0.1972 |
| SID91 | aef_hurts | 0.6143 | 0.5317 | -0.0827 |
| SID18 | aef_hurts | 0.8768 | 0.8204 | -0.0563 |
| SID17 | neutral | 0.6417 | 0.6057 | -0.0360 |
| SID55 | neutral | 0.5900 | 0.5635 | -0.0264 |
| SID59 | neutral | 0.7939 | 0.7845 | -0.0094 |
| SID85 | neutral | 0.9825 | 0.9807 | -0.0018 |
| SID93 | neutral | 0.0004 | 0.0000 | -0.0004 |
| SID04 | neutral | 0.7713 | 0.7712 | -0.0001 |
| SID72 | neutral | 0.9872 | 0.9881 | 0.0009 |

## Top AEF Improvements: permanent_water

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| SID23 | aef_helps | 0.0232 | 0.8691 | 0.8459 |
| SID100 | aef_helps | 0.1475 | 0.8208 | 0.6733 |
| SID89 | aef_helps | 0.3392 | 0.9294 | 0.5902 |
| SID33 | aef_helps | 0.1569 | 0.6429 | 0.4860 |
| SID67 | aef_helps | 0.0121 | 0.4873 | 0.4752 |

## Worst AEF Regressions: permanent_water

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| SID28 | aef_hurts | 0.9684 | 0.7712 | -0.1972 |
| SID91 | aef_hurts | 0.6143 | 0.5317 | -0.0827 |
| SID18 | aef_hurts | 0.8768 | 0.8204 | -0.0563 |
| SID17 | neutral | 0.6417 | 0.6057 | -0.0360 |
| SID55 | neutral | 0.5900 | 0.5635 | -0.0264 |

## Top AEF Improvements: not_permanent_water

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |

## Worst AEF Regressions: not_permanent_water

| Sample | Group | S1 IoU | S1+AEF IoU | Delta |
| --- | --- | ---: | ---: | ---: |
