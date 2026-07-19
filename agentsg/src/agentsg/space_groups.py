"""
The 230 standard-setting space groups: number, Hermann-Mauguin symbol,
Hall symbol, and crystal system.

This table is DATA, embedded as literals -- there is no runtime dependency on
any external package. Each Hall symbol was sourced from a verified Hall-symbol
table (transcription-free) and the whole table is validated operation-for-
operation against an independent oracle in tests/test_all_230.py.

Generators for any group are produced on demand by parsing its Hall symbol
(agentsg.hall) and closing the group -- exact rational arithmetic throughout,
no stored operation lists.
"""
from __future__ import annotations
from functools import lru_cache
from .hall import parse_hall
from .group import close_group

# (number, Hermann-Mauguin, Hall, crystal_system)
SPACE_GROUPS: tuple[tuple[int, str, str, str], ...] = (
    (1, 'P 1', 'P 1', 'triclinic'),
    (2, 'P -1', '-P 1', 'triclinic'),
    (3, 'P 1 2 1', 'P 2y', 'monoclinic'),
    (4, 'P 1 21 1', 'P 2yb', 'monoclinic'),
    (5, 'C 1 2 1', 'C 2y', 'monoclinic'),
    (6, 'P 1 m 1', 'P -2y', 'monoclinic'),
    (7, 'P 1 c 1', 'P -2yc', 'monoclinic'),
    (8, 'C 1 m 1', 'C -2y', 'monoclinic'),
    (9, 'C 1 c 1', 'C -2yc', 'monoclinic'),
    (10, 'P 1 2/m 1', '-P 2y', 'monoclinic'),
    (11, 'P 1 21/m 1', '-P 2yb', 'monoclinic'),
    (12, 'C 1 2/m 1', '-C 2y', 'monoclinic'),
    (13, 'P 1 2/c 1', '-P 2yc', 'monoclinic'),
    (14, 'P 1 21/c 1', '-P 2ybc', 'monoclinic'),
    (15, 'C 1 2/c 1', '-C 2yc', 'monoclinic'),
    (16, 'P 2 2 2', 'P 2 2', 'orthorhombic'),
    (17, 'P 2 2 21', 'P 2c 2', 'orthorhombic'),
    (18, 'P 21 21 2', 'P 2 2ab', 'orthorhombic'),
    (19, 'P 21 21 21', 'P 2ac 2ab', 'orthorhombic'),
    (20, 'C 2 2 21', 'C 2c 2', 'orthorhombic'),
    (21, 'C 2 2 2', 'C 2 2', 'orthorhombic'),
    (22, 'F 2 2 2', 'F 2 2', 'orthorhombic'),
    (23, 'I 2 2 2', 'I 2 2', 'orthorhombic'),
    (24, 'I 21 21 21', 'I 2b 2c', 'orthorhombic'),
    (25, 'P m m 2', 'P 2 -2', 'orthorhombic'),
    (26, 'P m c 21', 'P 2c -2', 'orthorhombic'),
    (27, 'P c c 2', 'P 2 -2c', 'orthorhombic'),
    (28, 'P m a 2', 'P 2 -2a', 'orthorhombic'),
    (29, 'P c a 21', 'P 2c -2ac', 'orthorhombic'),
    (30, 'P n c 2', 'P 2 -2bc', 'orthorhombic'),
    (31, 'P m n 21', 'P 2ac -2', 'orthorhombic'),
    (32, 'P b a 2', 'P 2 -2ab', 'orthorhombic'),
    (33, 'P n a 21', 'P 2c -2n', 'orthorhombic'),
    (34, 'P n n 2', 'P 2 -2n', 'orthorhombic'),
    (35, 'C m m 2', 'C 2 -2', 'orthorhombic'),
    (36, 'C m c 21', 'C 2c -2', 'orthorhombic'),
    (37, 'C c c 2', 'C 2 -2c', 'orthorhombic'),
    (38, 'A m m 2', 'A 2 -2', 'orthorhombic'),
    (39, 'A b m 2', 'A 2 -2b', 'orthorhombic'),
    (40, 'A m a 2', 'A 2 -2a', 'orthorhombic'),
    (41, 'A b a 2', 'A 2 -2ab', 'orthorhombic'),
    (42, 'F m m 2', 'F 2 -2', 'orthorhombic'),
    (43, 'F d d 2', 'F 2 -2d', 'orthorhombic'),
    (44, 'I m m 2', 'I 2 -2', 'orthorhombic'),
    (45, 'I b a 2', 'I 2 -2c', 'orthorhombic'),
    (46, 'I m a 2', 'I 2 -2a', 'orthorhombic'),
    (47, 'P m m m', '-P 2 2', 'orthorhombic'),
    (48, 'P n n n', 'P 2 2 -1n', 'orthorhombic'),
    (49, 'P c c m', '-P 2 2c', 'orthorhombic'),
    (50, 'P b a n', 'P 2 2 -1ab', 'orthorhombic'),
    (51, 'P m m a', '-P 2a 2a', 'orthorhombic'),
    (52, 'P n n a', '-P 2a 2bc', 'orthorhombic'),
    (53, 'P m n a', '-P 2ac 2', 'orthorhombic'),
    (54, 'P c c a', '-P 2a 2ac', 'orthorhombic'),
    (55, 'P b a m', '-P 2 2ab', 'orthorhombic'),
    (56, 'P c c n', '-P 2ab 2ac', 'orthorhombic'),
    (57, 'P b c m', '-P 2c 2b', 'orthorhombic'),
    (58, 'P n n m', '-P 2 2n', 'orthorhombic'),
    (59, 'P m m n', 'P 2 2ab -1ab', 'orthorhombic'),
    (60, 'P b c n', '-P 2n 2ab', 'orthorhombic'),
    (61, 'P b c a', '-P 2ac 2ab', 'orthorhombic'),
    (62, 'P n m a', '-P 2ac 2n', 'orthorhombic'),
    (63, 'C m c m', '-C 2c 2', 'orthorhombic'),
    (64, 'C m c a', '-C 2ac 2', 'orthorhombic'),
    (65, 'C m m m', '-C 2 2', 'orthorhombic'),
    (66, 'C c c m', '-C 2 2c', 'orthorhombic'),
    (67, 'C m m a', '-C 2a 2', 'orthorhombic'),
    (68, 'C c c a', 'C 2 2 -1ac', 'orthorhombic'),
    (69, 'F m m m', '-F 2 2', 'orthorhombic'),
    (70, 'F d d d', 'F 2 2 -1d', 'orthorhombic'),
    (71, 'I m m m', '-I 2 2', 'orthorhombic'),
    (72, 'I b a m', '-I 2 2c', 'orthorhombic'),
    (73, 'I b c a', '-I 2b 2c', 'orthorhombic'),
    (74, 'I m m a', '-I 2b 2', 'orthorhombic'),
    (75, 'P 4', 'P 4', 'tetragonal'),
    (76, 'P 41', 'P 4w', 'tetragonal'),
    (77, 'P 42', 'P 4c', 'tetragonal'),
    (78, 'P 43', 'P 4cw', 'tetragonal'),
    (79, 'I 4', 'I 4', 'tetragonal'),
    (80, 'I 41', 'I 4bw', 'tetragonal'),
    (81, 'P -4', 'P -4', 'tetragonal'),
    (82, 'I -4', 'I -4', 'tetragonal'),
    (83, 'P 4/m', '-P 4', 'tetragonal'),
    (84, 'P 42/m', '-P 4c', 'tetragonal'),
    (85, 'P 4/n', 'P 4ab -1ab', 'tetragonal'),
    (86, 'P 42/n', 'P 4n -1n', 'tetragonal'),
    (87, 'I 4/m', '-I 4', 'tetragonal'),
    (88, 'I 41/a', 'I 4bw -1bw', 'tetragonal'),
    (89, 'P 4 2 2', 'P 4 2', 'tetragonal'),
    (90, 'P 4 21 2', 'P 4ab 2ab', 'tetragonal'),
    (91, 'P 41 2 2', 'P 4w 2c', 'tetragonal'),
    (92, 'P 41 21 2', 'P 4abw 2nw', 'tetragonal'),
    (93, 'P 42 2 2', 'P 4c 2', 'tetragonal'),
    (94, 'P 42 21 2', 'P 4n 2n', 'tetragonal'),
    (95, 'P 43 2 2', 'P 4cw 2c', 'tetragonal'),
    (96, 'P 43 21 2', 'P 4nw 2abw', 'tetragonal'),
    (97, 'I 4 2 2', 'I 4 2', 'tetragonal'),
    (98, 'I 41 2 2', 'I 4bw 2bw', 'tetragonal'),
    (99, 'P 4 m m', 'P 4 -2', 'tetragonal'),
    (100, 'P 4 b m', 'P 4 -2ab', 'tetragonal'),
    (101, 'P 42 c m', 'P 4c -2c', 'tetragonal'),
    (102, 'P 42 n m', 'P 4n -2n', 'tetragonal'),
    (103, 'P 4 c c', 'P 4 -2c', 'tetragonal'),
    (104, 'P 4 n c', 'P 4 -2n', 'tetragonal'),
    (105, 'P 42 m c', 'P 4c -2', 'tetragonal'),
    (106, 'P 42 b c', 'P 4c -2ab', 'tetragonal'),
    (107, 'I 4 m m', 'I 4 -2', 'tetragonal'),
    (108, 'I 4 c m', 'I 4 -2c', 'tetragonal'),
    (109, 'I 41 m d', 'I 4bw -2', 'tetragonal'),
    (110, 'I 41 c d', 'I 4bw -2c', 'tetragonal'),
    (111, 'P -4 2 m', 'P -4 2', 'tetragonal'),
    (112, 'P -4 2 c', 'P -4 2c', 'tetragonal'),
    (113, 'P -4 21 m', 'P -4 2ab', 'tetragonal'),
    (114, 'P -4 21 c', 'P -4 2n', 'tetragonal'),
    (115, 'P -4 m 2', 'P -4 -2', 'tetragonal'),
    (116, 'P -4 c 2', 'P -4 -2c', 'tetragonal'),
    (117, 'P -4 b 2', 'P -4 -2ab', 'tetragonal'),
    (118, 'P -4 n 2', 'P -4 -2n', 'tetragonal'),
    (119, 'I -4 m 2', 'I -4 -2', 'tetragonal'),
    (120, 'I -4 c 2', 'I -4 -2c', 'tetragonal'),
    (121, 'I -4 2 m', 'I -4 2', 'tetragonal'),
    (122, 'I -4 2 d', 'I -4 2bw', 'tetragonal'),
    (123, 'P 4/m m m', '-P 4 2', 'tetragonal'),
    (124, 'P 4/m c c', '-P 4 2c', 'tetragonal'),
    (125, 'P 4/n b m', 'P 4 2 -1ab', 'tetragonal'),
    (126, 'P 4/n n c', 'P 4 2 -1n', 'tetragonal'),
    (127, 'P 4/m b m', '-P 4 2ab', 'tetragonal'),
    (128, 'P 4/m n c', '-P 4 2n', 'tetragonal'),
    (129, 'P 4/n m m', 'P 4ab 2ab -1ab', 'tetragonal'),
    (130, 'P 4/n c c', 'P 4ab 2n -1ab', 'tetragonal'),
    (131, 'P 42/m m c', '-P 4c 2', 'tetragonal'),
    (132, 'P 42/m c m', '-P 4c 2c', 'tetragonal'),
    (133, 'P 42/n b c', 'P 4n 2c -1n', 'tetragonal'),
    (134, 'P 42/n n m', 'P 4n 2 -1n', 'tetragonal'),
    (135, 'P 42/m b c', '-P 4c 2ab', 'tetragonal'),
    (136, 'P 42/m n m', '-P 4n 2n', 'tetragonal'),
    (137, 'P 42/n m c', 'P 4n 2n -1n', 'tetragonal'),
    (138, 'P 42/n c m', 'P 4n 2ab -1n', 'tetragonal'),
    (139, 'I 4/m m m', '-I 4 2', 'tetragonal'),
    (140, 'I 4/m c m', '-I 4 2c', 'tetragonal'),
    (141, 'I 41/a m d', 'I 4bw 2bw -1bw', 'tetragonal'),
    (142, 'I 41/a c d', 'I 4bw 2aw -1bw', 'tetragonal'),
    (143, 'P 3', 'P 3', 'trigonal'),
    (144, 'P 31', 'P 31', 'trigonal'),
    (145, 'P 32', 'P 32', 'trigonal'),
    (146, 'R 3', 'R 3', 'trigonal'),
    (147, 'P -3', '-P 3', 'trigonal'),
    (148, 'R -3', '-R 3', 'trigonal'),
    (149, 'P 3 1 2', 'P 3 2', 'trigonal'),
    (150, 'P 3 2 1', 'P 3 2"', 'trigonal'),
    (151, 'P 31 1 2', 'P 31 2 (0 0 4)', 'trigonal'),
    (152, 'P 31 2 1', 'P 31 2"', 'trigonal'),
    (153, 'P 32 1 2', 'P 32 2 (0 0 2)', 'trigonal'),
    (154, 'P 32 2 1', 'P 32 2"', 'trigonal'),
    (155, 'R 3 2', 'R 3 2"', 'trigonal'),
    (156, 'P 3 m 1', 'P 3 -2"', 'trigonal'),
    (157, 'P 3 1 m', 'P 3 -2', 'trigonal'),
    (158, 'P 3 c 1', 'P 3 -2"c', 'trigonal'),
    (159, 'P 3 1 c', 'P 3 -2c', 'trigonal'),
    (160, 'R 3 m', 'R 3 -2"', 'trigonal'),
    (161, 'R 3 c', 'R 3 -2"c', 'trigonal'),
    (162, 'P -3 1 m', '-P 3 2', 'trigonal'),
    (163, 'P -3 1 c', '-P 3 2c', 'trigonal'),
    (164, 'P -3 m 1', '-P 3 2"', 'trigonal'),
    (165, 'P -3 c 1', '-P 3 2"c', 'trigonal'),
    (166, 'R -3 m', '-R 3 2"', 'trigonal'),
    (167, 'R -3 c', '-R 3 2"c', 'trigonal'),
    (168, 'P 6', 'P 6', 'hexagonal'),
    (169, 'P 61', 'P 61', 'hexagonal'),
    (170, 'P 65', 'P 65', 'hexagonal'),
    (171, 'P 62', 'P 62', 'hexagonal'),
    (172, 'P 64', 'P 64', 'hexagonal'),
    (173, 'P 63', 'P 6c', 'hexagonal'),
    (174, 'P -6', 'P -6', 'hexagonal'),
    (175, 'P 6/m', '-P 6', 'hexagonal'),
    (176, 'P 63/m', '-P 6c', 'hexagonal'),
    (177, 'P 6 2 2', 'P 6 2', 'hexagonal'),
    (178, 'P 61 2 2', 'P 61 2 (0 0 5)', 'hexagonal'),
    (179, 'P 65 2 2', 'P 65 2 (0 0 1)', 'hexagonal'),
    (180, 'P 62 2 2', 'P 62 2 (0 0 4)', 'hexagonal'),
    (181, 'P 64 2 2', 'P 64 2 (0 0 2)', 'hexagonal'),
    (182, 'P 63 2 2', 'P 6c 2c', 'hexagonal'),
    (183, 'P 6 m m', 'P 6 -2', 'hexagonal'),
    (184, 'P 6 c c', 'P 6 -2c', 'hexagonal'),
    (185, 'P 63 c m', 'P 6c -2', 'hexagonal'),
    (186, 'P 63 m c', 'P 6c -2c', 'hexagonal'),
    (187, 'P -6 m 2', 'P -6 2', 'hexagonal'),
    (188, 'P -6 c 2', 'P -6c 2', 'hexagonal'),
    (189, 'P -6 2 m', 'P -6 -2', 'hexagonal'),
    (190, 'P -6 2 c', 'P -6c -2c', 'hexagonal'),
    (191, 'P 6/m m m', '-P 6 2', 'hexagonal'),
    (192, 'P 6/m c c', '-P 6 2c', 'hexagonal'),
    (193, 'P 63/m c m', '-P 6c 2', 'hexagonal'),
    (194, 'P 63/m m c', '-P 6c 2c', 'hexagonal'),
    (195, 'P 2 3', 'P 2 2 3', 'cubic'),
    (196, 'F 2 3', 'F 2 2 3', 'cubic'),
    (197, 'I 2 3', 'I 2 2 3', 'cubic'),
    (198, 'P 21 3', 'P 2ac 2ab 3', 'cubic'),
    (199, 'I 21 3', 'I 2b 2c 3', 'cubic'),
    (200, 'P m -3', '-P 2 2 3', 'cubic'),
    (201, 'P n -3', 'P 2 2 3 -1n', 'cubic'),
    (202, 'F m -3', '-F 2 2 3', 'cubic'),
    (203, 'F d -3', 'F 2 2 3 -1d', 'cubic'),
    (204, 'I m -3', '-I 2 2 3', 'cubic'),
    (205, 'P a -3', '-P 2ac 2ab 3', 'cubic'),
    (206, 'I a -3', '-I 2b 2c 3', 'cubic'),
    (207, 'P 4 3 2', 'P 4 2 3', 'cubic'),
    (208, 'P 42 3 2', 'P 4n 2 3', 'cubic'),
    (209, 'F 4 3 2', 'F 4 2 3', 'cubic'),
    (210, 'F 41 3 2', 'F 4d 2 3', 'cubic'),
    (211, 'I 4 3 2', 'I 4 2 3', 'cubic'),
    (212, 'P 43 3 2', 'P 4acd 2ab 3', 'cubic'),
    (213, 'P 41 3 2', 'P 4bd 2ab 3', 'cubic'),
    (214, 'I 41 3 2', 'I 4bd 2c 3', 'cubic'),
    (215, 'P -4 3 m', 'P -4 2 3', 'cubic'),
    (216, 'F -4 3 m', 'F -4 2 3', 'cubic'),
    (217, 'I -4 3 m', 'I -4 2 3', 'cubic'),
    (218, 'P -4 3 n', 'P -4n 2 3', 'cubic'),
    (219, 'F -4 3 c', 'F -4a 2 3', 'cubic'),
    (220, 'I -4 3 d', 'I -4bd 2c 3', 'cubic'),
    (221, 'P m -3 m', '-P 4 2 3', 'cubic'),
    (222, 'P n -3 n', 'P 4 2 3 -1n', 'cubic'),
    (223, 'P m -3 n', '-P 4n 2 3', 'cubic'),
    (224, 'P n -3 m', 'P 4n 2 3 -1n', 'cubic'),
    (225, 'F m -3 m', '-F 4 2 3', 'cubic'),
    (226, 'F m -3 c', '-F 4a 2 3', 'cubic'),
    (227, 'F d -3 m', 'F 4d 2 3 -1d', 'cubic'),
    (228, 'F d -3 c', 'F 4d 2 3 -1ad', 'cubic'),
    (229, 'I m -3 m', '-I 4 2 3', 'cubic'),
    (230, 'I a -3 d', '-I 4bd 2c 3', 'cubic'),
)

_BY_NUMBER = {r[0]: r for r in SPACE_GROUPS}

def _norm_hm(s: str) -> str:
    return s.replace(' ', '').replace('_', '').lower()

_BY_HM = {_norm_hm(r[1]): r for r in SPACE_GROUPS}
_BY_HALL = {r[2]: r for r in SPACE_GROUPS}


def _norm_hm_nodash(s: str) -> str:
    return _norm_hm(s).replace('-', '')

# Dash-insensitive fallback (e.g. "fm3m" for "F m -3 m"), but ONLY for symbols
# that stay unique once the bar is dropped. Centrosymmetric pairs like P1/P-1
# collide under dash-stripping, so they are deliberately excluded -- looking
# those up without the bar is genuinely ambiguous and should raise.
from collections import Counter as _Counter
_nodash_counts = _Counter(_norm_hm_nodash(r[1]) for r in SPACE_GROUPS)
_BY_HM_NODASH = {
    _norm_hm_nodash(r[1]): r
    for r in SPACE_GROUPS
    if _nodash_counts[_norm_hm_nodash(r[1])] == 1
}


class SpaceGroup:
    """A space group resolved from the standard-setting table.

    Attributes: number, hermann_mauguin, hall, crystal_system.
    Call .operations() for the closed, exact operation set (cached).
    """
    __slots__ = ('number', 'hermann_mauguin', 'hall', 'crystal_system')

    def __init__(self, row):
        self.number, self.hermann_mauguin, self.hall, self.crystal_system = row

    @lru_cache(maxsize=None)
    def operations(self):
        gens, cent = parse_hall(self.hall)
        return close_group(gens, cent)

    def order(self) -> int:
        return len(self.operations())

    def __repr__(self):
        return f'SpaceGroup(No. {self.number}, {self.hermann_mauguin!r}, Hall {self.hall!r})'


def space_group(key) -> SpaceGroup:
    """Look up a space group by number (1-230), Hermann-Mauguin symbol, or Hall symbol."""
    if isinstance(key, int):
        if key not in _BY_NUMBER:
            raise KeyError(f'space-group number {key} out of range 1..230')
        return SpaceGroup(_BY_NUMBER[key])
    if isinstance(key, str):
        if key in _BY_HALL:
            return SpaceGroup(_BY_HALL[key])
        nk = _norm_hm(key)
        if nk in _BY_HM:
            return SpaceGroup(_BY_HM[nk])
        ndk = _norm_hm_nodash(key)
        if ndk in _BY_HM_NODASH:
            return SpaceGroup(_BY_HM_NODASH[ndk])
        raise KeyError(f'unknown space-group symbol {key!r}')
    raise TypeError(f'space_group key must be int or str, got {type(key).__name__}')

