# Wall-incidence census

Measured cells from a `CellDatabase`, scored with the Selling-closure helpers in `manuscript/benchmark/evaluate.py` and `manuscript/appendix_z/test_z44_widened_complete.py`. No library code is changed.

Database `/home/sparksvc/Projects/agentsg/data/cod_cells.duckdb`. Seed `0`. Sample `None`. Loaded **528,439** cells. Closure pass: **full** on 528,439 obtuse cells. Metric batch on **cuda**, closures and sort paths on **20** processes (3764.27 closure cells/s). A full closure pass would take 0.04 h.

The reduced superbase is built from the stored Selling cell. `tau` is three times the largest conorm standard deviation, and never smaller than `1e-9 max|G|`. A parameter fixed by the reported crystal system (a 90° or 120° angle, or a length the system equates to another) has σ = 0. `tau_kind = esd` when every free parameter has an esd and at least one of those esds is positive; otherwise the nominal errors are `0.001 * length` and `0.03` degrees. A sort path imposes cosine-scaled zeros and ties in order. Each constraint set is replaced by the finest closed Table 4 pattern that contains it. A transition is inside the measurement when its cosine is within `3 σ(p)` of that slot, scaled by the edge lengths.

## Inventory

| source | n | with_esd | nominal | null_sg | non_P | not_obtuse |
| --- | --- | --- | --- | --- | --- | --- |
| cod | 528439 | 474931 | 53508 | 310 | 102906 | 0 |

| source | system | n |
| --- | --- | --- |
| cod | cubic | 18104 |
| cod | hexagonal | 9568 |
| cod | monoclinic | 249549 |
| cod | orthorhombic | 87631 |
| cod | tetragonal | 19582 |
| cod | triclinic | 127803 |
| cod | trigonal | 15892 |
| cod | unknown | 310 |

## Wall proximity

Histogram of `log10(min p / lam1)` in half-decade bins. `r=0` is a conorm at or below `1e-9 max|G|`.

| source | n | r=0 | <-6 | [-6.0,-5.5) | [-5.5,-5.0) | [-5.0,-4.5) | [-4.5,-4.0) | [-4.0,-3.5) | [-3.5,-3.0) | [-3.0,-2.5) | [-2.5,-2.0) | [-2.0,-1.5) | [-1.5,-1.0) | [-1.0,-0.5) | [-0.5,0.0) | >=0 | r<1e-02 | frac_r<1e-02 | r<1e-03 | frac_r<1e-03 | r<1e-04 | frac_r<1e-04 | r<1e-05 | frac_r<1e-05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | 528439 | 358829 | 13 | 14 | 19 | 55 | 189 | 522 | 1330 | 3423 | 8903 | 23400 | 53611 | 71520 | 6611 | 0 | 373297 | 0.7064145530515348 | 360971 | 0.6830892496579548 | 359119 | 0.6795845878142983 | 358875 | 0.6791228505087626 |

Near-zero conorms at `1e-3 lam1`, by reported system. Cubic, hexagonal, tetragonal and orthorhombic rows are expected to fill the two- and three-zero columns. On this database they do: hexagonal cells sit in the two-zero column, orthorhombic and tetragonal cells mostly in the three-zero column, and cubic cells in the two- and three-zero columns. A minority of cubic and tetragonal cells have no near-zero, and 132 triclinic cells have three.

| source | system | n | z0 | z1 | z2 | z3 | zmore |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cod | triclinic | 127803 | 125779 | 1659 | 233 | 132 | 0 |
| cod | monoclinic | 249549 | 25527 | 31742 | 191091 | 1189 | 0 |
| cod | orthorhombic | 87631 | 3391 | 1633 | 6582 | 76025 | 0 |
| cod | tetragonal | 19582 | 4848 | 4472 | 47 | 10215 | 0 |
| cod | trigonal | 15892 | 4444 | 13 | 11396 | 39 | 0 |
| cod | hexagonal | 9568 | 1 | 0 | 9564 | 3 | 0 |
| cod | cubic | 18104 | 3428 | 1 | 9886 | 4789 | 0 |
| cod | unknown | 310 | 50 | 12 | 99 | 149 | 0 |

## Closure

| source | system | tau_kind | n | n_1 | n_2 | n_3 | n_6 | n_16 | n_other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | triclinic | esd | 124500 | 120674 | 3469 | 31 | 151 | 26 | 84 |
| cod | monoclinic | esd | 237633 | 21703 | 29622 | 267 | 183257 | 2431 | 64 |
| cod | orthorhombic | esd | 77561 | 2775 | 1145 | 15 | 4673 | 68851 | 3 |
| cod | tetragonal | esd | 13280 | 3626 | 2303 | 18 | 0 | 7326 | 0 |
| cod | trigonal | esd | 10990 | 3640 | 1 | 3659 | 3662 | 13 | 5 |
| cod | hexagonal | esd | 4660 | 0 | 0 | 0 | 4643 | 0 | 0 |
| cod | cubic | esd | 6304 | 1671 | 0 | 2439 | 0 | 2194 | 0 |
| cod | unknown | esd | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| cod | triclinic | nominal | 3303 | 2429 | 500 | 25 | 159 | 140 | 19 |
| cod | monoclinic | nominal | 11916 | 3220 | 1857 | 166 | 5578 | 838 | 44 |
| cod | orthorhombic | nominal | 10070 | 545 | 510 | 21 | 1761 | 7107 | 0 |
| cod | tetragonal | nominal | 6302 | 1208 | 2138 | 67 | 1 | 2869 | 0 |
| cod | trigonal | nominal | 4902 | 794 | 12 | 2593 | 1338 | 34 | 17 |
| cod | hexagonal | nominal | 4908 | 0 | 0 | 0 | 4782 | 3 | 0 |
| cod | cubic | nominal | 11800 | 1756 | 1 | 7448 | 0 | 2595 | 0 |
| cod | unknown | nominal | 307 | 45 | 11 | 9 | 63 | 169 | 0 |

| source | system | tau_kind | n | walk_misses | walk2_misses | step2_singular | beyond_theorem | frac_walk_misses | frac_walk2_misses | frac_singular | frac_beyond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | triclinic | esd | 124500 | 79 | 1 | 65 | 809 | 0.0006345381526104418 | 8.032128514056226e-06 | 0.0005220883534136547 | 0.006497991967871486 |
| cod | monoclinic | esd | 237633 | 65 | 1 | 289 | 5118 | 0.000273531033147753 | 4.208169740734662e-06 | 0.0012161610550723175 | 0.021537412733080002 |
| cod | orthorhombic | esd | 77561 | 2 | 0 | 99 | 2551 | 2.5786155413158674e-05 | 0.0 | 0.0012764146929513545 | 0.03289024122948389 |
| cod | tetragonal | esd | 13280 | 0 | 0 | 7 | 137 | 0.0 | 0.0 | 0.0005271084337349398 | 0.010316265060240963 |
| cod | trigonal | esd | 10990 | 5 | 0 | 10 | 79 | 0.00045495905368516835 | 0.0 | 0.0009099181073703367 | 0.007188353048225659 |
| cod | hexagonal | esd | 4660 | 0 | 0 | 17 | 120 | 0.0 | 0.0 | 0.003648068669527897 | 0.02575107296137339 |
| cod | cubic | esd | 6304 | 0 | 0 | 0 | 4 | 0.0 | 0.0 | 0.0 | 0.0006345177664974619 |
| cod | unknown | esd | 3 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cod | triclinic | nominal | 3303 | 17 | 0 | 31 | 373 | 0.0051468362095065095 | 0.0 | 0.009385407205570693 | 0.11292764153799577 |
| cod | monoclinic | nominal | 11916 | 42 | 0 | 213 | 2377 | 0.003524672708962739 | 0.0 | 0.01787512588116818 | 0.19947969117153408 |
| cod | orthorhombic | nominal | 10070 | 0 | 0 | 126 | 3186 | 0.0 | 0.0 | 0.012512413108242304 | 0.3163853028798411 |
| cod | tetragonal | nominal | 6302 | 0 | 0 | 19 | 875 | 0.0 | 0.0 | 0.0030149158997143763 | 0.13884481117105682 |
| cod | trigonal | nominal | 4902 | 17 | 0 | 114 | 380 | 0.00346797225622195 | 0.0 | 0.023255813953488372 | 0.07751937984496124 |
| cod | hexagonal | nominal | 4908 | 0 | 0 | 123 | 909 | 0.0 | 0.0 | 0.02506112469437653 | 0.1852078239608802 |
| cod | cubic | nominal | 11800 | 0 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 | 8.474576271186441e-05 |
| cod | unknown | nominal | 307 | 0 | 0 | 10 | 68 | 0.0 | 0.0 | 0.03257328990228013 | 0.22149837133550487 |

## Symmetry

Visible order counts the signed label swaps that fix the conorm idealised at `4 tau`. It is a lower bound on the holohedry (Table 4, column visible). `metric-above-reported` is a visible order larger than the reported system's holohedry. `reported-above-visible` requires a single closure member, so a symmetry that lives on another member is not flagged.

| source | n | metric_above_reported | reported_above_visible |
| --- | --- | --- | --- |
| cod | 527316 | 2819 | 13 |

Ten most common off-diagonal (reported, visible) pairs.

| reported | visible | n |
| --- | --- | --- |
| monoclinic | triclinic | 220329 |
| orthorhombic | triclinic | 75017 |
| trigonal | monoclinic | 11056 |
| tetragonal | monoclinic | 10109 |
| cubic | tetragonal | 9888 |
| hexagonal | monoclinic | 9427 |
| orthorhombic | monoclinic | 8827 |
| cubic | rhombohedral | 4788 |
| trigonal | rhombohedral | 4426 |
| tetragonal | orthorhombic | 4383 |

## Classic traps

| monoclinic_repeat_centred | monoclinic_repeat_primitive | orthorhombic_closure_gt1 | walk_misses | walk_misses_in_traps | frac_walk_misses_in_traps |
| --- | --- | --- | --- | --- | --- |
| 32565 | 191558 | 84086 | 227 | 109 | 0.4801762114537445 |

## Niggli

Cells whose closure has more than one member, and the number of those whose Niggli cell differs from the reduced cell's. A count of zero means every other Selling member reduced to the same Niggli cell.

| source | sort | n | niggli_setting_differs |
| --- | --- | --- | --- |
| cod | A1 | 127444 | 339 |
| cod | A2 | 106 | 0 |
| cod | C1 | 3435 | 0 |
| cod | C3 | 9885 | 0 |
| cod | C5 | 4900 | 0 |
| cod | H4 | 14409 | 0 |
| cod | M1A | 12764 | 68 |
| cod | M1B | 12945 | 88 |
| cod | M2A | 31700 | 62 |
| cod | M4 | 191295 | 0 |
| cod | O1A | 2770 | 2 |
| cod | O1B | 637 | 7 |
| cod | O2 | 1611 | 1 |
| cod | O4 | 4593 | 0 |
| cod | O5 | 78313 | 0 |
| cod | R1 | 4462 | 0 |
| cod | R3 | 6267 | 0 |
| cod | T1 | 4865 | 49 |
| cod | T2 | 4521 | 132 |
| cod | T5 | 10391 | 0 |
| cod | unlisted | 3 | 0 |

## Specialisation

Populations at the measurement band (every transition inside its own cosine band) and at cosine tolerances `1e-2`, `1e-3`, `1e-4`.

| source | sort | tolerance | n |
| --- | --- | --- | --- |
| cod | A1 | 1e-2 | 75439 |
| cod | A1 | 1e-3 | 120425 |
| cod | A1 | 1e-4 | 126628 |
| cod | A1 | 4tau | 122541 |
| cod | A2 | 1e-2 | 43606 |
| cod | A2 | 1e-3 | 6580 |
| cod | A2 | 1e-4 | 878 |
| cod | A2 | 4tau | 4522 |
| cod | C3 | 1e-2 | 10620 |
| cod | C3 | 1e-3 | 10044 |
| cod | C3 | 1e-4 | 9924 |
| cod | C3 | 4tau | 10158 |
| cod | H4 | 1e-2 | 15301 |
| cod | H4 | 1e-3 | 14890 |
| cod | H4 | 1e-4 | 14770 |
| cod | H4 | 4tau | 14892 |
| cod | M1A | 1e-2 | 10951 |
| cod | M1A | 1e-3 | 12521 |
| cod | M1A | 1e-4 | 12825 |
| cod | M1A | 4tau | 12604 |
| cod | M1B | 1e-2 | 10578 |
| cod | M1B | 1e-3 | 12594 |
| cod | M1B | 1e-4 | 12883 |
| cod | M1B | 4tau | 12595 |
| cod | M2A | 1e-2 | 28976 |
| cod | M2A | 1e-3 | 31278 |
| cod | M2A | 1e-4 | 31711 |
| cod | M2A | 4tau | 31417 |
| cod | M4 | 1e-2 | 174005 |
| cod | M4 | 1e-3 | 188687 |
| cod | M4 | 1e-4 | 190968 |
| cod | M4 | 4tau | 189725 |
| cod | O5 | 1e-2 | 89533 |
| cod | O5 | 1e-3 | 80143 |
| cod | O5 | 1e-4 | 78798 |
| cod | O5 | 4tau | 78852 |
| cod | T5 | 1e-2 | 18957 |
| cod | T5 | 1e-3 | 11522 |
| cod | T5 | 1e-4 | 10604 |
| cod | T5 | 4tau | 11598 |

Ten most crossed edges. The histogram columns are half-decade bins of `log10 t` in cosine units.

| source | edge | n | <-6 | [-6.0,-5.5) | [-5.5,-5.0) | [-5.0,-4.5) | [-4.5,-4.0) | [-4.0,-3.5) | [-3.5,-3.0) | [-3.0,-2.5) | [-2.5,-2.0) | [-2.0,-1.5) | [-1.5,-1.0) | [-1.0,-0.5) | [-0.5,0.0) | >=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | A1>A2 | 88066 | 8 | 21 | 53 | 125 | 379 | 1023 | 2985 | 8523 | 22043 | 38025 | 14881 | 0 | 0 | 0 |
| cod | A2>M4 | 40963 | 0 | 2 | 1 | 4 | 7 | 32 | 64 | 259 | 1959 | 12385 | 25338 | 912 | 0 | 0 |
| cod | M4>O4: s - t = 0 | 34969 | 0 | 1 | 8 | 10 | 50 | 130 | 267 | 783 | 2243 | 6406 | 16531 | 8540 | 0 | 0 |
| cod | M4>O5 | 81486 | 4 | 29 | 73 | 106 | 143 | 223 | 517 | 1337 | 3807 | 12298 | 39772 | 23177 | 0 | 0 |
| cod | M4>O5: r = 0 | 68251 | 0 | 9 | 24 | 29 | 74 | 278 | 640 | 1700 | 4501 | 12595 | 30355 | 18046 | 0 | 0 |
| cod | M4>T5 | 40992 | 1 | 2 | 4 | 9 | 43 | 113 | 328 | 985 | 2851 | 8275 | 19175 | 9206 | 0 | 0 |
| cod | O5>T5: r - s = 0 | 66481 | 2 | 2 | 20 | 33 | 38 | 77 | 207 | 498 | 1433 | 3983 | 15278 | 43470 | 1440 | 0 |
| cod | O5>T5: r - t = 0 | 31780 | 0 | 0 | 1 | 7 | 12 | 28 | 70 | 199 | 602 | 1788 | 6663 | 21516 | 894 | 0 |
| cod | O5>T5: s - t = 0 | 45255 | 0 | 0 | 3 | 2 | 13 | 19 | 72 | 213 | 632 | 2278 | 9800 | 31274 | 949 | 0 |
| cod | T5>C5: r - s = 0 | 81317 | 1 | 0 | 0 | 1 | 1 | 13 | 15 | 49 | 145 | 655 | 4968 | 51490 | 23979 | 0 |

Population of the ten most common sorts along the cosine thresholds.

| source | sort | t | n |
| --- | --- | --- | --- |
| cod | A1 | 1.000e-06 | 127509 |
| cod | A1 | 3.162e-06 | 127476 |
| cod | A1 | 1.000e-05 | 127402 |
| cod | A1 | 3.162e-05 | 127200 |
| cod | A1 | 1.000e-04 | 126628 |
| cod | A1 | 3.162e-04 | 125019 |
| cod | A1 | 1.000e-03 | 120425 |
| cod | A1 | 3.162e-03 | 107685 |
| cod | A1 | 1.000e-02 | 75439 |
| cod | A1 | 3.162e-02 | 21089 |
| cod | C3 | 1.000e-06 | 9889 |
| cod | C3 | 3.162e-06 | 9892 |
| cod | C3 | 1.000e-05 | 9899 |
| cod | C3 | 3.162e-05 | 9908 |
| cod | C3 | 1.000e-04 | 9924 |
| cod | C3 | 3.162e-04 | 9969 |
| cod | C3 | 1.000e-03 | 10044 |
| cod | C3 | 3.162e-03 | 10222 |
| cod | C3 | 1.000e-02 | 10620 |
| cod | C3 | 3.162e-02 | 11457 |
| cod | C3 | 1.000e-01 | 14928 |
| cod | C3 | 3.162e-01 | 37609 |
| cod | C3 | 1.000e+00 | 37852 |
| cod | C5 | 1.000e-06 | 4906 |
| cod | C5 | 3.162e-06 | 4906 |
| cod | C5 | 1.000e-05 | 4907 |
| cod | C5 | 3.162e-05 | 4910 |
| cod | C5 | 1.000e-04 | 4913 |
| cod | C5 | 3.162e-04 | 4932 |
| cod | C5 | 1.000e-03 | 4968 |
| cod | C5 | 3.162e-03 | 5068 |
| cod | C5 | 1.000e-02 | 5430 |
| cod | C5 | 3.162e-02 | 7076 |
| cod | C5 | 1.000e-01 | 15876 |
| cod | C5 | 3.162e-01 | 97309 |
| cod | C5 | 1.000e+00 | 122809 |
| cod | H4 | 1.000e-06 | 14667 |
| cod | H4 | 3.162e-06 | 14677 |
| cod | H4 | 1.000e-05 | 14697 |
| cod | H4 | 3.162e-05 | 14733 |
| cod | H4 | 1.000e-04 | 14770 |
| cod | H4 | 3.162e-04 | 14811 |
| cod | H4 | 1.000e-03 | 14890 |
| cod | H4 | 3.162e-03 | 15005 |
| cod | H4 | 1.000e-02 | 15301 |
| cod | H4 | 3.162e-02 | 16759 |
| cod | H4 | 1.000e-01 | 30066 |
| cod | H4 | 3.162e-01 | 42656 |
| cod | H4 | 1.000e+00 | 40650 |
| cod | M2A | 1.000e-06 | 31785 |
| cod | M2A | 3.162e-06 | 31780 |
| cod | M2A | 1.000e-05 | 31764 |
| cod | M2A | 3.162e-05 | 31742 |
| cod | M2A | 1.000e-04 | 31711 |
| cod | M2A | 3.162e-04 | 31582 |
| cod | M2A | 1.000e-03 | 31278 |
| cod | M2A | 3.162e-03 | 30465 |
| cod | M2A | 1.000e-02 | 28976 |
| cod | M2A | 3.162e-02 | 28681 |
| cod | M2A | 1.000e-01 | 8872 |
| cod | M4 | 1.000e-06 | 191540 |
| cod | M4 | 3.162e-06 | 191499 |
| cod | M4 | 1.000e-05 | 191398 |
| cod | M4 | 3.162e-05 | 191253 |
| cod | M4 | 1.000e-04 | 190968 |
| cod | M4 | 3.162e-04 | 190312 |
| cod | M4 | 1.000e-03 | 188687 |
| cod | M4 | 3.162e-03 | 184382 |
| cod | M4 | 1.000e-02 | 174005 |
| cod | M4 | 3.162e-02 | 151960 |
| cod | M4 | 1.000e-01 | 71275 |
| cod | M4 | 3.162e-01 | 11316 |
| cod | M4 | 1.000e+00 | 11316 |
| cod | O4 | 1.000e-06 | 4637 |
| cod | O4 | 3.162e-06 | 4637 |
| cod | O4 | 1.000e-05 | 4643 |
| cod | O4 | 3.162e-05 | 4654 |
| cod | O4 | 1.000e-04 | 4711 |
| cod | O4 | 3.162e-04 | 4910 |
| cod | O4 | 1.000e-03 | 5320 |
| cod | O4 | 3.162e-03 | 6502 |
| cod | O4 | 1.000e-02 | 9934 |
| cod | O4 | 3.162e-02 | 20206 |
| cod | O4 | 1.000e-01 | 42936 |
| cod | O4 | 3.162e-01 | 10875 |
| cod | O4 | 1.000e+00 | 10871 |
| cod | O5 | 1.000e-06 | 78475 |
| cod | O5 | 3.162e-06 | 78506 |
| cod | O5 | 1.000e-05 | 78567 |
| cod | O5 | 3.162e-05 | 78639 |
| cod | O5 | 1.000e-04 | 78798 |
| cod | O5 | 3.162e-04 | 79216 |
| cod | O5 | 1.000e-03 | 80143 |
| cod | O5 | 3.162e-03 | 82688 |
| cod | O5 | 1.000e-02 | 89533 |
| cod | O5 | 3.162e-02 | 109781 |
| cod | O5 | 1.000e-01 | 154148 |
| cod | O5 | 3.162e-01 | 83358 |
| cod | O5 | 1.000e+00 | 80087 |
| cod | R3 | 1.000e-06 | 6293 |
| cod | R3 | 3.162e-06 | 6297 |
| cod | R3 | 1.000e-05 | 6312 |
| cod | R3 | 3.162e-05 | 6323 |
| cod | R3 | 1.000e-04 | 6343 |
| cod | R3 | 3.162e-04 | 6370 |
| cod | R3 | 1.000e-03 | 6492 |
| cod | R3 | 3.162e-03 | 6831 |
| cod | R3 | 1.000e-02 | 7785 |
| cod | R3 | 3.162e-02 | 11048 |
| cod | R3 | 1.000e-01 | 28876 |
| cod | R3 | 3.162e-01 | 34243 |
| cod | R3 | 1.000e+00 | 33706 |
| cod | T5 | 1.000e-06 | 10414 |
| cod | T5 | 3.162e-06 | 10418 |
| cod | T5 | 1.000e-05 | 10446 |
| cod | T5 | 3.162e-05 | 10496 |
| cod | T5 | 1.000e-04 | 10604 |
| cod | T5 | 3.162e-04 | 10831 |
| cod | T5 | 1.000e-03 | 11522 |
| cod | T5 | 3.162e-03 | 13429 |
| cod | T5 | 1.000e-02 | 18957 |
| cod | T5 | 3.162e-02 | 35598 |
| cod | T5 | 1.000e-01 | 90634 |
| cod | T5 | 3.162e-01 | 180208 |
| cod | T5 | 1.000e+00 | 160671 |

Fraction of cells whose first specialisation falls inside the cosine band of that edge (`ambiguous_esd`).

| source | sort_exact | n | ambiguous_esd | frac |
| --- | --- | --- | --- | --- |
| cod | A1 | 127523 | 4982 | 0.03906746234012688 |
| cod | A2 | 110 | 36 | 0.32727272727272727 |
| cod | C1 | 3435 | 0 | 0.0 |
| cod | C3 | 9885 | 0 | 0.0 |
| cod | C5 | 4900 | 0 | 0.0 |
| cod | H4 | 14656 | 35 | 0.0023881004366812226 |
| cod | M1A | 12904 | 359 | 0.02782083075015499 |
| cod | M1B | 12964 | 431 | 0.03324591175563098 |
| cod | M2A | 31787 | 507 | 0.01594991663258565 |
| cod | M4 | 191551 | 1990 | 0.010388878157775214 |
| cod | O1A | 2795 | 54 | 0.01932021466905188 |
| cod | O1B | 637 | 65 | 0.10204081632653061 |
| cod | O2 | 1614 | 70 | 0.04337050805452292 |
| cod | O4 | 4641 | 112 | 0.024132730015082957 |
| cod | O5 | 78474 | 786 | 0.010016056273415399 |
| cod | R1 | 4463 | 31 | 0.0069460004481290616 |
| cod | R3 | 6292 | 85 | 0.0135092180546726 |
| cod | T1 | 4868 | 60 | 0.012325390304026294 |
| cod | T2 | 4525 | 130 | 0.028729281767955802 |
| cod | T5 | 10412 | 31 | 0.002977333845562812 |
| cod | unlisted | 3 | 0 | 0.0 |

## Spot checks

- COD9000001: `COD9000001` (cod), formula `- Pb S -`, sg 225 F m -3 m, cell 5.93 5.93 5.93 90.0 90.0 90.0. lam1=17.5825 tau=0.0527474 (nominal) visible=16 sort=C3 n_closure=3 n_walk_tau=3 n_walk_2tau=3
- alpha-quartz: `COD1011097` (cod), formula `- O2 Si -`, sg 152 P 31 2 1, cell 4.913 4.913 5.404 90.0 90.0 120.0. lam1=24.1376 tau=0.184879 (nominal) visible=4 sort=H4 n_closure=6 n_walk_tau=6 n_walk_2tau=6
- NaCl: `COD1000041` (cod), formula `- Cl Na -`, sg 225 F m -3 m, cell 5.62 5.62 5.62 90.0 90.0 90.0. lam1=15.7922 tau=0.0473766 (nominal) visible=16 sort=C3 n_closure=3 n_walk_tau=3 n_walk_2tau=3
- Si: `COD1526655` (cod), formula `- Si -`, sg 227 F d -3 m :1, cell 5.381 5.381 5.381 90.0 90.0 90.0. lam1=14.4776 tau=0.0434327 (nominal) visible=16 sort=C3 n_closure=3 n_walk_tau=3 n_walk_2tau=3
- albite: `COD1521704` (cod), formula `- Al Na O8 Si3 -`, sg 12 C 1 2/m 1, cell 8.126 12.996 7.164 90.0 116.65 90.0. lam1=51.3229 tau=0.254666 (nominal) visible=4 sort=M1A n_closure=1 n_walk_tau=1 n_walk_2tau=1
- diopside: `COD1000007` (cod), formula `- Ca Mg O6 Si2 -`, sg 15 C 1 2/c 1, cell 9.7397 8.9174 5.2503 90.0 105.866 90.0. lam1=27.5657 tau=0.223636 (nominal) visible=4 sort=M1B n_closure=1 n_walk_tau=1 n_walk_2tau=1
- AMCSD feldspar: not in this database.
- 193L: not in this database.
- 1LYZ: not in this database.
- 2HHB: not in this database.

## Issues

- **triclinic-with-three-near-zeros** (132): Triclinic-reported cells with at least three conorms below 1e-3 lam1. Examples: COD1008070, COD1010618, COD1101054, COD1507446, COD1508808.
- **walk-misses-with-esd** (151): Flip walk at tau missed a closure member on a cell whose band came from the reported esds. Examples: COD7128791, COD1000252, COD1007093, COD1007211, COD1502144.
- **beyond-theorem** (16987): lam1 is not greater than 32 tau, so the widened-closure theorem's hypothesis fails. The construction was still run. Examples: COD1510704, COD1510933, COD1513798, COD1529963, COD1534599.
- **step2-singular** (1123): The idealised lattice at 4 tau was singular, so members_by('closure') returned None, or the flip search exceeded 5 seconds. Examples: COD2000630, COD7046135, COD9004102, COD9005493, COD1000036.
- **unlisted-sort** (3): A closed (zeros, equality classes) set matched none of the 24 Table 4 patterns. Examples: COD2107367, COD9002434, COD9009269.
- **path-longer-than-six** (11948): The monotone path changed sort more than six times. The first six steps are stored. Examples: COD1000507, COD1001380, COD1001564, COD1001646, COD1004000.

## Environment

```
git 0c9b910b018592527ed89a0c4b06ce669c8779ee
agentsg 0.5.0
duckdb 1.5.5
numpy 2.5.3
rows {'cod': 528439}
device cuda
workers 20
wall_seconds 103.614
closure_mode full
cells_per_second 3764.2743
schema_meta
  archive=cod
  cod_source=/home/sparksvc/Downloads/COD.csv
  schema_version=4
  search_key=sorted_linear_key
```

Conorm file `manuscript/census/conorms.npz` is 48,500,135 bytes. It is committed only when that is under 10 MB.
