# Wall-incidence census

Measured cells from a `CellDatabase`, scored with the Selling-closure helpers in `manuscript/benchmark/evaluate.py` and `manuscript/appendix_z/test_z44_widened_complete.py`. No library code is changed.

Database `/home/sparksvc/Projects/agentsg/data/cod_cells.duckdb`. Seed `0`. Sample `None`. Loaded **528,439** cells. Closure pass: **full** on 528,439 obtuse cells. Metric batch on **cuda**, closures and sort paths on **20** processes (3420.48 closure cells/s). A full closure pass would take 0.04 h.

The reduced superbase is built from the stored Selling cell. `tau` is three times the largest conorm standard deviation. Where all six esds are present that deviation comes from a numerical Jacobian through the stored change of basis (`tau_kind = esd`); otherwise the nominal errors are `0.001 * length` and `0.03` degrees (`tau_kind = nominal`). A sort path imposes cosine-scaled zeros and ties in order and closes them under transitivity, so the sort only moves up the specialisation diagram.

## Inventory

| source | n | with_esd | null_sg | non_P | not_obtuse |
| --- | --- | --- | --- | --- | --- |
| cod | 528439 | 127782 | 310 | 102906 | 0 |

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
| cod | monoclinic | esd | 2242 | 255 | 293 | 3 | 1652 | 32 | 4 |
| cod | orthorhombic | esd | 575 | 29 | 3 | 0 | 29 | 512 | 1 |
| cod | tetragonal | esd | 115 | 36 | 21 | 0 | 0 | 58 | 0 |
| cod | trigonal | esd | 286 | 113 | 0 | 104 | 66 | 1 | 1 |
| cod | hexagonal | esd | 32 | 0 | 0 | 0 | 32 | 0 | 0 |
| cod | cubic | esd | 29 | 9 | 0 | 5 | 0 | 15 | 0 |
| cod | unknown | esd | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| cod | triclinic | nominal | 3303 | 2429 | 500 | 25 | 159 | 140 | 19 |
| cod | monoclinic | nominal | 247307 | 22838 | 30828 | 1316 | 176710 | 12300 | 350 |
| cod | orthorhombic | nominal | 87056 | 3191 | 1661 | 53 | 6212 | 74639 | 4 |
| cod | tetragonal | nominal | 19467 | 4738 | 4391 | 126 | 1 | 10058 | 0 |
| cod | trigonal | nominal | 15606 | 4300 | 12 | 6109 | 4886 | 74 | 44 |
| cod | hexagonal | nominal | 9536 | 0 | 0 | 0 | 9222 | 3 | 0 |
| cod | cubic | nominal | 18075 | 3418 | 1 | 9882 | 0 | 4774 | 0 |
| cod | unknown | nominal | 307 | 45 | 11 | 9 | 63 | 169 | 0 |

| source | system | tau_kind | n | walk_misses | walk2_misses | step2_singular | beyond_theorem | frac_walk_misses | frac_walk2_misses | frac_singular | frac_beyond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | triclinic | esd | 124500 | 79 | 1 | 65 | 809 | 0.0006345381526104418 | 8.032128514056226e-06 | 0.0005220883534136547 | 0.006497991967871486 |
| cod | monoclinic | esd | 2242 | 5 | 1 | 3 | 76 | 0.0022301516503122213 | 0.00044603033006244426 | 0.0013380909901873326 | 0.03389830508474576 |
| cod | orthorhombic | esd | 575 | 0 | 0 | 1 | 13 | 0.0 | 0.0 | 0.0017391304347826088 | 0.022608695652173914 |
| cod | tetragonal | esd | 115 | 0 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 | 0.008695652173913044 |
| cod | trigonal | esd | 286 | 1 | 0 | 1 | 6 | 0.0034965034965034965 | 0.0 | 0.0034965034965034965 | 0.02097902097902098 |
| cod | hexagonal | esd | 32 | 1 | 1 | 0 | 0 | 0.03125 | 0.03125 | 0.0 | 0.0 |
| cod | cubic | esd | 29 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cod | unknown | esd | 3 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cod | triclinic | nominal | 3303 | 17 | 0 | 31 | 373 | 0.0051468362095065095 | 0.0 | 0.009385407205570693 | 0.11292764153799577 |
| cod | monoclinic | nominal | 247307 | 369 | 2 | 2965 | 72897 | 0.0014920726061130498 | 8.087114396276693e-06 | 0.011989147092480197 | 0.294763189072691 |
| cod | orthorhombic | nominal | 87056 | 4 | 0 | 1296 | 38235 | 4.594743613306377e-05 | 0.0 | 0.014886969307112663 | 0.43920005513692334 |
| cod | tetragonal | nominal | 19467 | 0 | 0 | 153 | 3401 | 0.0 | 0.0 | 0.00785945446139621 | 0.17470591256999024 |
| cod | trigonal | nominal | 15606 | 44 | 0 | 181 | 1285 | 0.002819428424964757 | 0.0 | 0.011598103293605024 | 0.08234012559272075 |
| cod | hexagonal | nominal | 9536 | 0 | 0 | 311 | 2050 | 0.0 | 0.0 | 0.03261325503355705 | 0.2149748322147651 |
| cod | cubic | nominal | 18075 | 0 | 0 | 0 | 1 | 0.0 | 0.0 | 0.0 | 5.532503457814661e-05 |
| cod | unknown | nominal | 307 | 0 | 0 | 10 | 68 | 0.0 | 0.0 | 0.03257328990228013 | 0.22149837133550487 |

## Symmetry

Visible order counts the signed label swaps that fix the conorm idealised at `4 tau`. It is a lower bound on the holohedry (Table 4, column visible). `metric-above-reported` is a visible order larger than the reported system's holohedry. `reported-above-visible` requires a single closure member, so a symmetry that lives on another member is not flagged.

| source | n | metric_above_reported | reported_above_visible |
| --- | --- | --- | --- |
| cod | 523422 | 4991 | 12 |

Ten most common off-diagonal (reported, visible) pairs.

| reported | visible | n |
| --- | --- | --- |
| monoclinic | triclinic | 206685 |
| orthorhombic | triclinic | 67119 |
| orthorhombic | monoclinic | 15631 |
| trigonal | monoclinic | 10919 |
| tetragonal | monoclinic | 9938 |
| cubic | tetragonal | 9888 |
| hexagonal | monoclinic | 9256 |
| cubic | rhombohedral | 4788 |
| trigonal | rhombohedral | 4384 |
| tetragonal | orthorhombic | 4328 |

## Classic traps

| monoclinic_repeat_centred | monoclinic_repeat_primitive | orthorhombic_closure_gt1 | walk_misses | walk_misses_in_traps | frac_walk_misses_in_traps |
| --- | --- | --- | --- | --- | --- |
| 34086 | 189401 | 83114 | 520 | 378 | 0.7269230769230769 |

## Niggli

Cells whose closure has more than one member, and the number of those whose Niggli cell differs from the reduced cell's. A count of zero means every other Selling member reduced to the same Niggli cell.

| source | n_closure_gt1 | niggli_differs |
| --- | --- | --- |
| cod | 361344 | 0 |

## Specialisation

Populations at each cell's own `4 tau` and at cosine tolerances `1e-2`, `1e-3`, `1e-4`.

| source | sort | tolerance | n |
| --- | --- | --- | --- |
| cod | A1 | 1e-2 | 110606 |
| cod | A1 | 1e-3 | 125026 |
| cod | A1 | 1e-4 | 127218 |
| cod | A1 | 4tau | 88101 |
| cod | C3 | 1e-2 | 10364 |
| cod | C3 | 1e-3 | 10005 |
| cod | C3 | 1e-4 | 9916 |
| cod | C3 | 4tau | 15180 |
| cod | C5 | 1e-2 | 5091 |
| cod | C5 | 1e-3 | 4930 |
| cod | C5 | 1e-4 | 4904 |
| cod | C5 | 4tau | 47008 |
| cod | H4 | 1e-2 | 15004 |
| cod | H4 | 1e-3 | 14788 |
| cod | H4 | 1e-4 | 14720 |
| cod | H4 | 4tau | 25772 |
| cod | M1A | 1e-2 | 12001 |
| cod | M1A | 1e-3 | 12698 |
| cod | M1A | 1e-4 | 12841 |
| cod | M1A | 4tau | 7753 |
| cod | M1B | 1e-2 | 11995 |
| cod | M1B | 1e-3 | 12780 |
| cod | M1B | 1e-4 | 12913 |
| cod | M1B | 4tau | 8611 |
| cod | M2A | 1e-2 | 31922 |
| cod | M2A | 1e-3 | 31794 |
| cod | M2A | 1e-4 | 31778 |
| cod | M2A | 4tau | 32739 |
| cod | M4 | 1e-2 | 181425 |
| cod | M4 | 1e-3 | 190156 |
| cod | M4 | 1e-4 | 191374 |
| cod | M4 | 4tau | 104334 |
| cod | O5 | 1e-2 | 79509 |
| cod | O5 | 1e-3 | 76781 |
| cod | O5 | 1e-4 | 76332 |
| cod | O5 | 4tau | 38980 |
| cod | T5 | 1e-2 | 14461 |
| cod | T5 | 1e-3 | 11003 |
| cod | T5 | 1e-4 | 10541 |
| cod | T5 | 4tau | 88978 |

Ten most crossed edges. The histogram columns are half-decade bins of `log10 t` in cosine units.

| source | edge | n | <-6 | [-6.0,-5.5) | [-5.5,-5.0) | [-5.0,-4.5) | [-4.5,-4.0) | [-4.0,-3.5) | [-3.5,-3.0) | [-3.0,-2.5) | [-2.5,-2.0) | [-2.0,-1.5) | [-1.5,-1.0) | [-1.0,-0.5) | [-0.5,0.0) | >=0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cod | A1>A2: r = 0 | 14042 | 3 | 7 | 8 | 26 | 53 | 205 | 574 | 1468 | 3640 | 5886 | 2172 | 0 | 0 | 0 |
| cod | A1>A2: s = 0 | 9899 | 0 | 2 | 2 | 16 | 51 | 124 | 356 | 956 | 2503 | 4258 | 1631 | 0 | 0 | 0 |
| cod | M4>O4: s - t = 0 | 29754 | 0 | 1 | 8 | 9 | 49 | 130 | 267 | 782 | 2227 | 6093 | 13103 | 7085 | 0 | 0 |
| cod | M4>O5: r = 0 | 66369 | 0 | 9 | 23 | 28 | 74 | 273 | 639 | 1696 | 4487 | 12471 | 29315 | 17354 | 0 | 0 |
| cod | O4>H4: r - s = 0 | 12301 | 4 | 2 | 4 | 4 | 21 | 16 | 30 | 43 | 119 | 405 | 2544 | 9108 | 1 | 0 |
| cod | O4>T5: r = 0 | 13499 | 0 | 0 | 0 | 0 | 1 | 1 | 22 | 45 | 112 | 513 | 2885 | 9920 | 0 | 0 |
| cod | O5>T5: r - s = 0 | 45773 | 2 | 2 | 20 | 33 | 38 | 77 | 205 | 487 | 1383 | 3607 | 11296 | 27330 | 1293 | 0 |
| cod | O5>T5: r - t = 0 | 26771 | 0 | 0 | 1 | 7 | 11 | 28 | 70 | 195 | 590 | 1655 | 5653 | 17778 | 783 | 0 |
| cod | O5>T5: s - t = 0 | 31203 | 0 | 0 | 3 | 2 | 12 | 16 | 67 | 205 | 595 | 2002 | 7105 | 20288 | 908 | 0 |
| cod | T5>C5: r - s = 0 | 41997 | 1 | 0 | 0 | 1 | 1 | 11 | 13 | 43 | 112 | 405 | 2267 | 21569 | 17574 | 0 |

Population of the ten most common sorts along the cosine thresholds.

| source | sort | t | n |
| --- | --- | --- | --- |
| cod | A1 | 1.000e-06 | 127517 |
| cod | A1 | 3.162e-06 | 127505 |
| cod | A1 | 1.000e-05 | 127486 |
| cod | A1 | 3.162e-05 | 127409 |
| cod | A1 | 1.000e-04 | 127218 |
| cod | A1 | 3.162e-04 | 126633 |
| cod | A1 | 1.000e-03 | 125026 |
| cod | A1 | 3.162e-03 | 120809 |
| cod | A1 | 1.000e-02 | 110606 |
| cod | A1 | 3.162e-02 | 94281 |
| cod | A1 | 1.000e-01 | 88074 |
| cod | A1 | 3.162e-01 | 88074 |
| cod | A1 | 1.000e+00 | 88074 |
| cod | C3 | 1.000e-06 | 9885 |
| cod | C3 | 3.162e-06 | 9888 |
| cod | C3 | 1.000e-05 | 9895 |
| cod | C3 | 3.162e-05 | 9902 |
| cod | C3 | 1.000e-04 | 9916 |
| cod | C3 | 3.162e-04 | 9949 |
| cod | C3 | 1.000e-03 | 10005 |
| cod | C3 | 3.162e-03 | 10130 |
| cod | C3 | 1.000e-02 | 10364 |
| cod | C3 | 3.162e-02 | 10849 |
| cod | C3 | 1.000e-01 | 12088 |
| cod | C3 | 3.162e-01 | 14987 |
| cod | C3 | 1.000e+00 | 15254 |
| cod | C5 | 1.000e-06 | 4901 |
| cod | C5 | 3.162e-06 | 4901 |
| cod | C5 | 1.000e-05 | 4901 |
| cod | C5 | 3.162e-05 | 4903 |
| cod | C5 | 1.000e-04 | 4904 |
| cod | C5 | 3.162e-04 | 4915 |
| cod | C5 | 1.000e-03 | 4930 |
| cod | C5 | 3.162e-03 | 4975 |
| cod | C5 | 1.000e-02 | 5091 |
| cod | C5 | 3.162e-02 | 5520 |
| cod | C5 | 1.000e-01 | 7856 |
| cod | C5 | 3.162e-01 | 29543 |
| cod | C5 | 1.000e+00 | 47127 |
| cod | H4 | 1.000e-06 | 14662 |
| cod | H4 | 3.162e-06 | 14667 |
| cod | H4 | 1.000e-05 | 14677 |
| cod | H4 | 3.162e-05 | 14692 |
| cod | H4 | 1.000e-04 | 14720 |
| cod | H4 | 3.162e-04 | 14747 |
| cod | H4 | 1.000e-03 | 14788 |
| cod | H4 | 3.162e-03 | 14850 |
| cod | H4 | 1.000e-02 | 15004 |
| cod | H4 | 3.162e-02 | 15535 |
| cod | H4 | 1.000e-01 | 18290 |
| cod | H4 | 3.162e-01 | 26265 |
| cod | H4 | 1.000e+00 | 25789 |
| cod | M1A | 1.000e-06 | 12904 |
| cod | M1A | 3.162e-06 | 12902 |
| cod | M1A | 1.000e-05 | 12894 |
| cod | M1A | 3.162e-05 | 12883 |
| cod | M1A | 1.000e-04 | 12841 |
| cod | M1A | 3.162e-04 | 12791 |
| cod | M1A | 1.000e-03 | 12698 |
| cod | M1A | 3.162e-03 | 12501 |
| cod | M1A | 1.000e-02 | 12001 |
| cod | M1A | 3.162e-02 | 10563 |
| cod | M1A | 1.000e-01 | 8235 |
| cod | M1A | 3.162e-01 | 7753 |
| cod | M1A | 1.000e+00 | 7753 |
| cod | M1B | 1.000e-06 | 12964 |
| cod | M1B | 3.162e-06 | 12962 |
| cod | M1B | 1.000e-05 | 12956 |
| cod | M1B | 3.162e-05 | 12943 |
| cod | M1B | 1.000e-04 | 12913 |
| cod | M1B | 3.162e-04 | 12877 |
| cod | M1B | 1.000e-03 | 12780 |
| cod | M1B | 3.162e-03 | 12554 |
| cod | M1B | 1.000e-02 | 11995 |
| cod | M1B | 3.162e-02 | 10723 |
| cod | M1B | 1.000e-01 | 8767 |
| cod | M1B | 3.162e-01 | 8611 |
| cod | M1B | 1.000e+00 | 8611 |
| cod | M2A | 1.000e-06 | 31771 |
| cod | M2A | 3.162e-06 | 31768 |
| cod | M2A | 1.000e-05 | 31770 |
| cod | M2A | 3.162e-05 | 31774 |
| cod | M2A | 1.000e-04 | 31778 |
| cod | M2A | 3.162e-04 | 31774 |
| cod | M2A | 1.000e-03 | 31794 |
| cod | M2A | 3.162e-03 | 31832 |
| cod | M2A | 1.000e-02 | 31922 |
| cod | M2A | 3.162e-02 | 32687 |
| cod | M2A | 1.000e-01 | 34067 |
| cod | M2A | 3.162e-01 | 32760 |
| cod | M2A | 1.000e+00 | 32760 |
| cod | M4 | 1.000e-06 | 191550 |
| cod | M4 | 3.162e-06 | 191540 |
| cod | M4 | 1.000e-05 | 191516 |
| cod | M4 | 3.162e-05 | 191483 |
| cod | M4 | 1.000e-04 | 191374 |
| cod | M4 | 3.162e-04 | 191011 |
| cod | M4 | 1.000e-03 | 190156 |
| cod | M4 | 3.162e-03 | 187805 |
| cod | M4 | 1.000e-02 | 181425 |
| cod | M4 | 3.162e-02 | 164830 |
| cod | M4 | 1.000e-01 | 127346 |
| cod | M4 | 3.162e-01 | 104337 |
| cod | M4 | 1.000e+00 | 104337 |
| cod | O5 | 1.000e-06 | 76327 |
| cod | O5 | 3.162e-06 | 76334 |
| cod | O5 | 1.000e-05 | 76333 |
| cod | O5 | 3.162e-05 | 76319 |
| cod | O5 | 1.000e-04 | 76332 |
| cod | O5 | 3.162e-04 | 76484 |
| cod | O5 | 1.000e-03 | 76781 |
| cod | O5 | 3.162e-03 | 77590 |
| cod | O5 | 1.000e-02 | 79509 |
| cod | O5 | 3.162e-02 | 84717 |
| cod | O5 | 1.000e-01 | 89978 |
| cod | O5 | 3.162e-01 | 41936 |
| cod | O5 | 1.000e+00 | 38952 |
| cod | T5 | 1.000e-06 | 10413 |
| cod | T5 | 3.162e-06 | 10415 |
| cod | T5 | 1.000e-05 | 10439 |
| cod | T5 | 3.162e-05 | 10480 |
| cod | T5 | 1.000e-04 | 10541 |
| cod | T5 | 3.162e-04 | 10652 |
| cod | T5 | 1.000e-03 | 11003 |
| cod | T5 | 3.162e-03 | 11893 |
| cod | T5 | 1.000e-02 | 14461 |
| cod | T5 | 3.162e-02 | 21834 |
| cod | T5 | 1.000e-01 | 46525 |
| cod | T5 | 3.162e-01 | 102921 |
| cod | T5 | 1.000e+00 | 88966 |

Fraction of cells whose exact sort does not survive `4 tau` (`ambiguous_esd`).

| source | sort_exact | n | ambiguous_esd | frac |
| --- | --- | --- | --- | --- |
| cod | A1 | 127523 | 39422 | 0.309136391082393 |
| cod | A2 | 105 | 51 | 0.4857142857142857 |
| cod | C1 | 3435 | 0 | 0.0 |
| cod | C3 | 9881 | 0 | 0.0 |
| cod | C5 | 4899 | 0 | 0.0 |
| cod | H4 | 14656 | 0 | 0.0 |
| cod | M1A | 12904 | 5152 | 0.3992560446373218 |
| cod | M1B | 12964 | 4355 | 0.335930268435668 |
| cod | M2A | 31773 | 3056 | 0.09618229314197589 |
| cod | M4 | 191550 | 94217 | 0.49186635343252416 |
| cod | O1A | 2795 | 1167 | 0.41753130590339893 |
| cod | O1B | 637 | 637 | 1.0 |
| cod | O2 | 1612 | 485 | 0.3008684863523573 |
| cod | O4 | 4641 | 2217 | 0.4776987718164189 |
| cod | O5 | 76329 | 56845 | 0.7447366007677292 |
| cod | R1 | 4463 | 1945 | 0.4358055119874524 |
| cod | R3 | 6292 | 2681 | 0.4260966306420852 |
| cod | T1 | 4868 | 2075 | 0.426253081347576 |
| cod | T2 | 4516 | 2030 | 0.44951284322409213 |
| cod | T5 | 10412 | 3624 | 0.34805993084902037 |
| cod | unlisted | 2184 | 614 | 0.28113553113553114 |

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
- **walk-misses-with-esd** (86): Flip walk at tau missed a closure member on a cell whose band came from the reported esds. Examples: COD1007093, COD1504291, COD1505444, COD1505700, COD1507213.
- **beyond-theorem** (119215): lam1 is not greater than 32 tau, so the widened-closure theorem's hypothesis fails. The construction was still run. Examples: COD1008817, COD1008856, COD1502380, COD1502400, COD1502525.
- **step2-singular** (5017): The idealised lattice at 4 tau was singular, so members_by('closure') returned None. Examples: COD1518871, COD1550148, COD2000630, COD2100164, COD2225510.
- **unlisted-sort** (2184): A closed (zeros, equality classes) set matched none of the 24 Table 4 patterns. Examples: COD1000031, COD1000073, COD1000074, COD1000103, COD1000200.
- **non-codimension-one-jump** (8495): A sort change was not one of the 37 codimension-one edges of Table C2. Examples: COD1000038, COD1000042, COD1000103, COD1000191, COD1000192.

## Environment

```
git 9a389d33d771c883396b266fa73a5935b1536412
agentsg 0.5.0
duckdb 1.5.5
numpy 2.5.3
rows {'cod': 528439}
device cuda
workers 20
wall_seconds 122.541
closure_mode full
cells_per_second 3420.4819
schema_meta
  archive=cod
  cod_source=/home/sparksvc/Downloads/COD.csv
  schema_version=4
  search_key=sorted_linear_key
```

Conorm file `manuscript/census/conorms.npz` is 33,236,869 bytes. It is committed only when that is under 10 MB.
