# Progressive metric perturbation vs Le Page / Kurlin

Ideal cells from each crystal system are randomly perturbed (seed=0) at increasing edge/angle noise. Assignment uses `lattice_symmetry(..., max_delta=3.0°)` on the Niggli-reduced cell (Le Page gate). Spectrum tables report, for every candidate holohedry, the **max Le Page δ** over that system's two-folds and the **Kurlin** root-invariant distance to its Reynolds-symmetrised metric, both scored in the **conventional (unreduced) setting** so the fixed candidate operators and the cell always share one basis (avoiding the Niggli reduction-flip that otherwise injects a spurious baseline).

> Note: the `triclinic` column is 0 at every level by construction — triclinic symmetrisation is just inversion, which leaves the metric unchanged. It carries no discriminating information; rank candidates by symmetry order among those passing the δ gate, not by raw Kurlin.

## Ideal cells (conventional setting)

| system | order | a | b | c | α | β | γ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cubic | 48 | 5 | 5 | 5 | 90 | 90 | 90 |
| tetragonal | 16 | 5 | 5 | 8 | 90 | 90 | 90 |
| hexagonal | 24 | 5 | 5 | 8 | 90 | 90 | 120 |
| trigonal | 12 | 5 | 5 | 5 | 70 | 70 | 70 |
| orthorhombic | 8 | 5 | 6 | 7 | 90 | 90 | 90 |
| monoclinic | 4 | 5 | 6 | 7 | 90 | 95 | 90 |
| triclinic | 2 | 5 | 6 | 7 | 80 | 85 | 95 |

## Noise ladder

| level | σ_edge (fraction) | σ_angle (°) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 0.002 | 0.15 |
| 2 | 0.005 | 0.4 |
| 3 | 0.01 | 0.8 |
| 4 | 0.02 | 1.5 |
| 5 | 0.04 | 3 |

## Le Page assignment (`max_delta=3.0°`)

| base | level | σ_edge | σ_angle° | assigned | order | cell |
| --- | --- | --- | --- | --- | --- | --- |
| cubic | 0 | 0 | 0 | cubic | 48 | 5 / 5 / 5 / 90 / 90 / 90 |
| cubic | 1 | 0.002 | 0.15 | cubic | 48 | 4.986 / 5.004 / 4.988 / 90.23 / 89.82 / 90.06 |
| cubic | 2 | 0.005 | 0.4 | cubic | 48 | 5.001 / 5.022 / 4.987 / 89.32 / 90.53 / 89.88 |
| cubic | 3 | 0.01 | 0.8 | cubic | 48 | 4.966 / 4.953 / 4.993 / 90.4 / 89.52 / 89.74 |
| cubic | 4 | 0.02 | 1.5 | cubic | 48 | 5.087 / 4.98 / 5.019 / 92.07 / 88.76 / 90.04 |
| cubic | 5 | 0.04 | 3 | cubic | 48 | 4.879 / 4.83 / 5.038 / 86.96 / 88.64 / 90.9 |
| tetragonal | 0 | 0 | 0 | tetragonal | 16 | 5 / 5 / 8 / 90 / 90 / 90 |
| tetragonal | 1 | 0.002 | 0.15 | tetragonal | 16 | 4.994 / 4.998 / 7.986 / 89.87 / 89.86 / 89.85 |
| tetragonal | 2 | 0.005 | 0.4 | tetragonal | 16 | 5.004 / 5.014 / 7.948 / 90.21 / 89.85 / 89.87 |
| tetragonal | 3 | 0.01 | 0.8 | tetragonal | 16 | 5.108 / 4.943 / 8.043 / 89.83 / 89.31 / 89.95 |
| tetragonal | 4 | 0.02 | 1.5 | tetragonal | 16 | 4.939 / 5.022 / 7.919 / 91.67 / 89.98 / 89.06 |
| tetragonal | 5 | 0.04 | 3 | monoclinic | 4 | 4.827 / 4.998 / 7.451 / 86.24 / 90.03 / 91.25 |
| hexagonal | 0 | 0 | 0 | hexagonal | 24 | 5 / 5 / 8 / 90 / 90 / 120 |
| hexagonal | 1 | 0.002 | 0.15 | hexagonal | 24 | 5.004 / 5.001 / 8.024 / 90.02 / 90.08 / 120 |
| hexagonal | 2 | 0.005 | 0.4 | hexagonal | 24 | 4.992 / 5.05 / 8.027 / 90.6 / 90.34 / 120.1 |
| hexagonal | 3 | 0.01 | 0.8 | hexagonal | 24 | 4.899 / 5.06 / 8.152 / 90.81 / 90.69 / 118.7 |
| hexagonal | 4 | 0.02 | 1.5 | hexagonal | 24 | 4.94 / 4.778 / 7.827 / 86.87 / 89.22 / 122.3 |
| hexagonal | 5 | 0.04 | 3 | hexagonal | 24 | 5.122 / 4.932 / 8.405 / 87.71 / 89.36 / 120.7 |
| trigonal | 0 | 0 | 0 | trigonal | 12 | 5 / 5 / 5 / 70 / 70 / 70 |
| trigonal | 1 | 0.002 | 0.15 | trigonal | 12 | 4.988 / 4.998 / 5.01 / 69.76 / 70.04 / 70.02 |
| trigonal | 2 | 0.005 | 0.4 | trigonal | 12 | 5.032 / 4.995 / 4.989 / 69.8 / 69.92 / 70.61 |
| trigonal | 3 | 0.01 | 0.8 | trigonal | 12 | 5.013 / 4.991 / 5.035 / 69.03 / 70.68 / 70.07 |
| trigonal | 4 | 0.02 | 1.5 | trigonal | 12 | 5.127 / 5.168 / 5.18 / 70.03 / 70.39 / 71.22 |
| trigonal | 5 | 0.04 | 3 | triclinic | 2 | 4.88 / 5.067 / 5.021 / 68.52 / 66.3 / 76.84 |
| orthorhombic | 0 | 0 | 0 | orthorhombic | 8 | 5 / 6 / 7 / 90 / 90 / 90 |
| orthorhombic | 1 | 0.002 | 0.15 | orthorhombic | 8 | 4.991 / 6.002 / 7.002 / 90.01 / 90.1 / 89.82 |
| orthorhombic | 2 | 0.005 | 0.4 | orthorhombic | 8 | 5 / 6.002 / 7.013 / 90.13 / 89.84 / 90.15 |
| orthorhombic | 3 | 0.01 | 0.8 | orthorhombic | 8 | 5.091 / 5.953 / 7.103 / 88.57 / 89.12 / 89.37 |
| orthorhombic | 4 | 0.02 | 1.5 | orthorhombic | 8 | 4.918 / 6.003 / 7.029 / 87.63 / 90.72 / 90.12 |
| orthorhombic | 5 | 0.04 | 3 | monoclinic | 4 | 4.954 / 5.973 / 6.776 / 91.24 / 84.48 / 88.66 |
| monoclinic | 0 | 0 | 0 | monoclinic | 4 | 5 / 6 / 7 / 90 / 95 / 90 |
| monoclinic | 1 | 0.002 | 0.15 | monoclinic | 4 | 5.012 / 6.003 / 7.012 / 90.13 / 95.1 / 89.77 |
| monoclinic | 2 | 0.005 | 0.4 | monoclinic | 4 | 4.985 / 5.963 / 7.033 / 89.74 / 94.43 / 90.42 |
| monoclinic | 3 | 0.01 | 0.8 | monoclinic | 4 | 5.016 / 5.95 / 7.134 / 89.81 / 93.94 / 89.08 |
| monoclinic | 4 | 0.02 | 1.5 | monoclinic | 4 | 4.856 / 6.014 / 7.086 / 89.03 / 93.33 / 88.23 |
| monoclinic | 5 | 0.04 | 3 | triclinic | 2 | 4.938 / 5.94 / 7.363 / 93.03 / 97.08 / 87.85 |
| triclinic | 0 | 0 | 0 | triclinic | 2 | 5 / 6 / 7 / 80 / 85 / 95 |
| triclinic | 1 | 0.002 | 0.15 | triclinic | 2 | 5.004 / 6.005 / 6.98 / 80.14 / 84.81 / 95.13 |
| triclinic | 2 | 0.005 | 0.4 | triclinic | 2 | 5.036 / 5.98 / 7.063 / 79.28 / 85.15 / 94.5 |
| triclinic | 3 | 0.01 | 0.8 | triclinic | 2 | 5.063 / 6.053 / 6.986 / 78.63 / 85.48 / 94.63 |
| triclinic | 4 | 0.02 | 1.5 | triclinic | 2 | 5.022 / 6.033 / 7.003 / 79.36 / 83.82 / 96.72 |
| triclinic | 5 | 0.04 | 3 | triclinic | 2 | 5.238 / 6.292 / 7.245 / 80.04 / 84.21 / 104.2 |

## Kurlin distance to each holohedry (Å)

Rows = noise level for a fixed base ideal cell; columns = candidate holohedry. Values are Kurlin root distances (0 = exact match).

### Base: cubic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 4.284 | 0 | 0 | 0 | 0 |
| 1 | 0.3882 | 0.3881 | 4.024 | 0.1716 | 0.3879 | 0.2785 | 0 |
| 2 | 0.7659 | 0.7652 | 3.787 | 0.6384 | 0.7651 | 0.5372 | 0 |
| 3 | 0.7055 | 0.7052 | 3.841 | 0.5865 | 0.7052 | 0.5347 | 0 |
| 4 | 1.217 | 1.216 | 3.433 | 0.7577 | 1.212 | 0.7798 | 0 |
| 5 | 1.528 | 1.525 | 3.177 | 1.09 | 1.523 | 1.067 | 0 |

### Base: tetragonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.466 | 0 | 4.284 | 2.466 | 0 | 0 | 0 |
| 1 | 2.487 | 0.3612 | 4.046 | 2.474 | 0.3612 | 0.2659 | 0 |
| 2 | 2.482 | 0.5515 | 3.944 | 2.472 | 0.5515 | 0.4037 | 0 |
| 3 | 2.601 | 0.78 | 3.656 | 2.514 | 0.775 | 0.3461 | 0 |
| 4 | 2.761 | 1.267 | 3.275 | 2.55 | 1.267 | 1.181 | 0 |
| 5 | 2.749 | 1.744 | 2.754 | 2.372 | 1.745 | 1.618 | 0 |

### Base: hexagonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5.449 | 4.097 | 0 | 4.939 | 4.097 | 4.097 | 0 |
| 1 | 5.469 | 4.106 | 0.269 | 4.812 | 4.106 | 3.896 | 0 |
| 2 | 5.472 | 4.133 | 0.8185 | 4.5 | 4.133 | 3.726 | 0 |
| 3 | 5.53 | 4.107 | 1.05 | 4.448 | 4.105 | 3.531 | 0 |
| 4 | 5.154 | 3.838 | 1.706 | 3.904 | 3.835 | 3.307 | 0 |
| 5 | 5.561 | 4.008 | 1.527 | 4.372 | 4.01 | 3.476 | 0 |

### Base: trigonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3.35 | 3.35 | 1.84 | 0 | 3.35 | 0.9442 | 0 |
| 1 | 3.376 | 3.376 | 1.889 | 0.3881 | 3.376 | 1.022 | 0 |
| 2 | 3.415 | 3.414 | 2.004 | 0.7371 | 3.414 | 1.169 | 0 |
| 3 | 3.476 | 3.476 | 2.104 | 0.9139 | 3.476 | 1.298 | 0 |
| 4 | 3.527 | 3.527 | 2.196 | 1.001 | 3.527 | 1.399 | 0 |
| 5 | 4.298 | 4.295 | 3.685 | 2.952 | 4.286 | 3.111 | 0 |

### Base: orthorhombic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.417 | 0.7078 | 4.775 | 1.417 | 0 | 0 | 0 |
| 1 | 1.485 | 0.8171 | 4.497 | 1.468 | 0.3842 | 0.249 | 0 |
| 2 | 1.468 | 0.7866 | 4.514 | 1.434 | 0.3358 | 0.1923 | 0 |
| 3 | 1.817 | 1.282 | 4.019 | 1.596 | 1.138 | 0.7586 | 0 |
| 4 | 2.054 | 1.629 | 3.553 | 1.711 | 1.472 | 0.9375 | 0 |
| 5 | 2.657 | 2.432 | 3.283 | 2.074 | 2.255 | 1.263 | 0 |

### Base: monoclinic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.335 | 2.007 | 3.152 | 2.198 | 1.788 | 0 | 0 |
| 1 | 2.361 | 2.032 | 3.176 | 2.005 | 1.819 | 0.3457 | 0 |
| 2 | 2.277 | 1.898 | 3.289 | 1.901 | 1.689 | 0.4691 | 0 |
| 3 | 2.398 | 1.972 | 3.359 | 1.83 | 1.79 | 0.7911 | 0 |
| 4 | 2.573 | 2.161 | 3.617 | 2.168 | 1.942 | 1.287 | 0 |
| 5 | 3.067 | 2.632 | 3.332 | 1.891 | 2.473 | 1.562 | 0 |

### Base: triclinic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4.078 | 3.874 | 3.113 | 3.102 | 3.838 | 2.787 | 0 |
| 1 | 4.086 | 3.888 | 3.157 | 3.127 | 3.848 | 2.8 | 0 |
| 2 | 4.112 | 3.903 | 3.011 | 3.05 | 3.882 | 2.792 | 0 |
| 3 | 4.133 | 3.964 | 2.976 | 3.019 | 3.946 | 2.86 | 0 |
| 4 | 4.436 | 4.237 | 3.33 | 3.475 | 4.195 | 3.104 | 0 |
| 5 | 5.249 | 4.983 | 3.924 | 4.67 | 4.928 | 3.969 | 0 |

## Max Le Page δ to each holohedry (°)

Same layout; entry is the largest Le Page delta among the two-folds of that candidate holohedry (empty two-fold set → 0).

### Base: cubic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 26.57 | 0 | 0 | 0 | 0 |
| 1 | 0.3554 | 0.3554 | 26.6 | 0.3554 | 0.2918 | 0.2398 | 0 |
| 2 | 0.8896 | 0.8896 | 26.77 | 0.8896 | 0.8652 | 0.6941 | 0 |
| 3 | 0.7033 | 0.647 | 26.84 | 0.647 | 0.631 | 0.4808 | 0 |
| 4 | 2.642 | 2.642 | 27.1 | 2.642 | 2.41 | 2.066 | 0 |
| 5 | 3.347 | 3.347 | 26.25 | 3.347 | 3.347 | 3.19 | 0 |

### Base: tetragonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 25.99 | 0 | 26.57 | 25.99 | 0 | 0 | 0 |
| 1 | 25.97 | 0.2048 | 26.7 | 25.97 | 0.2048 | 0.1986 | 0 |
| 2 | 25.61 | 0.278 | 26.71 | 25.61 | 0.2559 | 0.2428 | 0 |
| 3 | 26.86 | 1.98 | 27.37 | 26.86 | 0.715 | 0.1776 | 0 |
| 4 | 26.14 | 1.919 | 27.73 | 26.14 | 1.919 | 1.919 | 0 |
| 5 | 24.31 | 3.962 | 26.58 | 24.31 | 3.962 | 3.962 | 0 |

### Base: hexagonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 36.17 | 30 | 0 | 36.17 | 30 | 30 | 0 |
| 1 | 36.28 | 29.95 | 0.1128 | 36.22 | 29.95 | 29.95 | 0 |
| 2 | 36.65 | 30.14 | 1.208 | 36.22 | 30.14 | 30.14 | 0 |
| 3 | 37.22 | 28.74 | 2.658 | 36.67 | 28.74 | 28.74 | 0 |
| 4 | 39 | 32.59 | 4.801 | 39 | 32.59 | 32.59 | 0 |
| 5 | 39.33 | 30.84 | 3.892 | 39.33 | 30.84 | 30.84 | 0 |

### Base: trigonal

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 24.68 | 24.68 | 43.93 | 0 | 24.68 | 24.68 | 0 |
| 1 | 24.83 | 24.83 | 43.97 | 0.324 | 24.83 | 24.83 | 0 |
| 2 | 24.96 | 24.96 | 43.74 | 0.9721 | 24.96 | 24.42 | 0 |
| 3 | 25.38 | 25.38 | 44.32 | 1.555 | 25.38 | 25.38 | 0 |
| 4 | 24.61 | 24.61 | 43.21 | 1.151 | 24.61 | 23.97 | 0 |
| 5 | 29.47 | 29.47 | 42.02 | 9.724 | 29.41 | 22.12 | 0 |

### Base: orthorhombic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 18.92 | 10.39 | 30.96 | 18.92 | 0 | 0 | 0 |
| 1 | 19.04 | 10.51 | 31.15 | 19.04 | 0.2019 | 0.1774 | 0 |
| 2 | 19.03 | 10.41 | 30.87 | 19.03 | 0.2133 | 0.1941 | 0 |
| 3 | 18.79 | 9.072 | 30.8 | 18.75 | 1.673 | 1.559 | 0 |
| 4 | 20.1 | 11.53 | 31.39 | 20.1 | 2.48 | 2.376 | 0 |
| 5 | 17.83 | 11.8 | 32.49 | 17.83 | 5.716 | 1.921 | 0 |

### Base: monoclinic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 18.99 | 11.07 | 31.33 | 18.99 | 5 | 0 | 0 |
| 1 | 18.95 | 11.02 | 31.46 | 18.95 | 5.106 | 0.2497 | 0 |
| 2 | 19.41 | 10.8 | 30.86 | 19.41 | 4.447 | 0.4754 | 0 |
| 3 | 19.84 | 10.23 | 31.57 | 19.83 | 4.046 | 0.954 | 0 |
| 4 | 21.28 | 12.58 | 33.19 | 21.21 | 3.794 | 2.07 | 0 |
| 5 | 22.68 | 12.78 | 33.25 | 22.68 | 7.604 | 3.524 | 0 |

### Base: triclinic

| level | cubic | tetragonal | hexagonal | trigonal | orthorhombic | monoclinic | triclinic |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 21.62 | 15 | 27.9 | 21.62 | 11.64 | 11.64 | 0 |
| 1 | 21.47 | 15.04 | 27.78 | 21.47 | 11.62 | 11.6 | 0 |
| 2 | 21.71 | 14.85 | 28.14 | 21.71 | 12.18 | 12.04 | 0 |
| 3 | 21.2 | 15.24 | 28.39 | 21.2 | 12.68 | 12.68 | 0 |
| 4 | 22.49 | 16.24 | 26.98 | 22.49 | 13.31 | 13.31 | 0 |
| 5 | 25.77 | 18.41 | 21.82 | 25.77 | 18.41 | 18.41 | 0 |

## Combined spectra (selected bases)

### cubic

**level 0** → assigned `cubic` (order 48)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 0 | 0 |
| tetragonal | 0 | 0 |
| hexagonal | 26.57 | 4.284 |
| trigonal | 0 | 0 |
| orthorhombic | 0 | 0 |
| monoclinic | 0 | 0 |
| triclinic | 0 | 0 |

**level 2** → assigned `cubic` (order 48)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 0.8896 | 0.7659 |
| tetragonal | 0.8896 | 0.7652 |
| hexagonal | 26.77 | 3.787 |
| trigonal | 0.8896 | 0.6384 |
| orthorhombic | 0.8652 | 0.7651 |
| monoclinic | 0.6941 | 0.5372 |
| triclinic | 0 | 0 |

**level 4** → assigned `cubic` (order 48)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 2.642 | 1.217 |
| tetragonal | 2.642 | 1.216 |
| hexagonal | 27.1 | 3.433 |
| trigonal | 2.642 | 0.7577 |
| orthorhombic | 2.41 | 1.212 |
| monoclinic | 2.066 | 0.7798 |
| triclinic | 0 | 0 |

**level 5** → assigned `cubic` (order 48)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 3.347 | 1.528 |
| tetragonal | 3.347 | 1.525 |
| hexagonal | 26.25 | 3.177 |
| trigonal | 3.347 | 1.09 |
| orthorhombic | 3.347 | 1.523 |
| monoclinic | 3.19 | 1.067 |
| triclinic | 0 | 0 |

### tetragonal

**level 0** → assigned `tetragonal` (order 16)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 25.99 | 2.466 |
| tetragonal | 0 | 0 |
| hexagonal | 26.57 | 4.284 |
| trigonal | 25.99 | 2.466 |
| orthorhombic | 0 | 0 |
| monoclinic | 0 | 0 |
| triclinic | 0 | 0 |

**level 2** → assigned `tetragonal` (order 16)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 25.61 | 2.482 |
| tetragonal | 0.278 | 0.5515 |
| hexagonal | 26.71 | 3.944 |
| trigonal | 25.61 | 2.472 |
| orthorhombic | 0.2559 | 0.5515 |
| monoclinic | 0.2428 | 0.4037 |
| triclinic | 0 | 0 |

**level 4** → assigned `tetragonal` (order 16)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 26.14 | 2.761 |
| tetragonal | 1.919 | 1.267 |
| hexagonal | 27.73 | 3.275 |
| trigonal | 26.14 | 2.55 |
| orthorhombic | 1.919 | 1.267 |
| monoclinic | 1.919 | 1.181 |
| triclinic | 0 | 0 |

**level 5** → assigned `monoclinic` (order 4)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 24.31 | 2.749 |
| tetragonal | 3.962 | 1.744 |
| hexagonal | 26.58 | 2.754 |
| trigonal | 24.31 | 2.372 |
| orthorhombic | 3.962 | 1.745 |
| monoclinic | 3.962 | 1.618 |
| triclinic | 0 | 0 |

### monoclinic

**level 0** → assigned `monoclinic` (order 4)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 18.99 | 2.335 |
| tetragonal | 11.07 | 2.007 |
| hexagonal | 31.33 | 3.152 |
| trigonal | 18.99 | 2.198 |
| orthorhombic | 5 | 1.788 |
| monoclinic | 0 | 0 |
| triclinic | 0 | 0 |

**level 2** → assigned `monoclinic` (order 4)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 19.41 | 2.277 |
| tetragonal | 10.8 | 1.898 |
| hexagonal | 30.86 | 3.289 |
| trigonal | 19.41 | 1.901 |
| orthorhombic | 4.447 | 1.689 |
| monoclinic | 0.4754 | 0.4691 |
| triclinic | 0 | 0 |

**level 4** → assigned `monoclinic` (order 4)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 21.28 | 2.573 |
| tetragonal | 12.58 | 2.161 |
| hexagonal | 33.19 | 3.617 |
| trigonal | 21.21 | 2.168 |
| orthorhombic | 3.794 | 1.942 |
| monoclinic | 2.07 | 1.287 |
| triclinic | 0 | 0 |

**level 5** → assigned `triclinic` (order 2)

| holohedry | max Le Page δ (°) | Kurlin (Å) |
| --- | --- | --- |
| cubic | 22.68 | 3.067 |
| tetragonal | 12.78 | 2.632 |
| hexagonal | 33.25 | 3.332 |
| trigonal | 22.68 | 1.891 |
| orthorhombic | 7.604 | 2.473 |
| monoclinic | 3.524 | 1.562 |
| triclinic | 0 | 0 |

