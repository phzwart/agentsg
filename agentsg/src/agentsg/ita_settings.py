"""ITA alternate settings.
Generated from the ITA setting list. There is no runtime gemmi import.
Each row is (number, Hermann-Mauguin, Hall, tag, hm_2016).
tag is '', '1', '2', 'H', or 'R'. hm_2016 is empty when it matches
the Hermann-Mauguin symbol.
"""
from __future__ import annotations
import re

ITA_SETTINGS: tuple[tuple[int, str, str, str, str], ...] = (
    (1, 'P 1', 'P 1', '', ''),
    (2, 'P -1', '-P 1', '', ''),
    (3, 'P 1 2 1', 'P 2y', '', ''),
    (3, 'P 1 1 2', 'P 2', '', ''),
    (3, 'P 2 1 1', 'P 2x', '', ''),
    (4, 'P 1 21 1', 'P 2yb', '', ''),
    (4, 'P 1 1 21', 'P 2c', '', ''),
    (4, 'P 21 1 1', 'P 2xa', '', ''),
    (5, 'C 1 2 1', 'C 2y', '', ''),
    (5, 'A 1 2 1', 'A 2y', '', ''),
    (5, 'I 1 2 1', 'I 2y', '', ''),
    (5, 'A 1 1 2', 'A 2', '', ''),
    (5, 'B 1 1 2', 'B 2', '', ''),
    (5, 'I 1 1 2', 'I 2', '', ''),
    (5, 'B 2 1 1', 'B 2x', '', ''),
    (5, 'C 2 1 1', 'C 2x', '', ''),
    (5, 'I 2 1 1', 'I 2x', '', ''),
    (6, 'P 1 m 1', 'P -2y', '', ''),
    (6, 'P 1 1 m', 'P -2', '', ''),
    (6, 'P m 1 1', 'P -2x', '', ''),
    (7, 'P 1 c 1', 'P -2yc', '', ''),
    (7, 'P 1 n 1', 'P -2yac', '', ''),
    (7, 'P 1 a 1', 'P -2ya', '', ''),
    (7, 'P 1 1 a', 'P -2a', '', ''),
    (7, 'P 1 1 n', 'P -2ab', '', ''),
    (7, 'P 1 1 b', 'P -2b', '', ''),
    (7, 'P b 1 1', 'P -2xb', '', ''),
    (7, 'P n 1 1', 'P -2xbc', '', ''),
    (7, 'P c 1 1', 'P -2xc', '', ''),
    (8, 'C 1 m 1', 'C -2y', '', ''),
    (8, 'A 1 m 1', 'A -2y', '', ''),
    (8, 'I 1 m 1', 'I -2y', '', ''),
    (8, 'A 1 1 m', 'A -2', '', ''),
    (8, 'B 1 1 m', 'B -2', '', ''),
    (8, 'I 1 1 m', 'I -2', '', ''),
    (8, 'B m 1 1', 'B -2x', '', ''),
    (8, 'C m 1 1', 'C -2x', '', ''),
    (8, 'I m 1 1', 'I -2x', '', ''),
    (9, 'C 1 c 1', 'C -2yc', '', ''),
    (9, 'A 1 n 1', 'A -2yab', '', ''),
    (9, 'I 1 a 1', 'I -2ya', '', ''),
    (9, 'A 1 a 1', 'A -2ya', '', ''),
    (9, 'C 1 n 1', 'C -2yac', '', ''),
    (9, 'I 1 c 1', 'I -2yc', '', ''),
    (9, 'A 1 1 a', 'A -2a', '', ''),
    (9, 'B 1 1 n', 'B -2ab', '', ''),
    (9, 'I 1 1 b', 'I -2b', '', ''),
    (9, 'B 1 1 b', 'B -2b', '', ''),
    (9, 'A 1 1 n', 'A -2ab', '', ''),
    (9, 'I 1 1 a', 'I -2a', '', ''),
    (9, 'B b 1 1', 'B -2xb', '', ''),
    (9, 'C n 1 1', 'C -2xac', '', ''),
    (9, 'I c 1 1', 'I -2xc', '', ''),
    (9, 'C c 1 1', 'C -2xc', '', ''),
    (9, 'B n 1 1', 'B -2xab', '', ''),
    (9, 'I b 1 1', 'I -2xb', '', ''),
    (10, 'P 1 2/m 1', '-P 2y', '', ''),
    (10, 'P 1 1 2/m', '-P 2', '', ''),
    (10, 'P 2/m 1 1', '-P 2x', '', ''),
    (11, 'P 1 21/m 1', '-P 2yb', '', ''),
    (11, 'P 1 1 21/m', '-P 2c', '', ''),
    (11, 'P 21/m 1 1', '-P 2xa', '', ''),
    (12, 'C 1 2/m 1', '-C 2y', '', ''),
    (12, 'A 1 2/m 1', '-A 2y', '', ''),
    (12, 'I 1 2/m 1', '-I 2y', '', ''),
    (12, 'A 1 1 2/m', '-A 2', '', ''),
    (12, 'B 1 1 2/m', '-B 2', '', ''),
    (12, 'I 1 1 2/m', '-I 2', '', ''),
    (12, 'B 2/m 1 1', '-B 2x', '', ''),
    (12, 'C 2/m 1 1', '-C 2x', '', ''),
    (12, 'I 2/m 1 1', '-I 2x', '', ''),
    (13, 'P 1 2/c 1', '-P 2yc', '', ''),
    (13, 'P 1 2/n 1', '-P 2yac', '', ''),
    (13, 'P 1 2/a 1', '-P 2ya', '', ''),
    (13, 'P 1 1 2/a', '-P 2a', '', ''),
    (13, 'P 1 1 2/n', '-P 2ab', '', ''),
    (13, 'P 1 1 2/b', '-P 2b', '', ''),
    (13, 'P 2/b 1 1', '-P 2xb', '', ''),
    (13, 'P 2/n 1 1', '-P 2xbc', '', ''),
    (13, 'P 2/c 1 1', '-P 2xc', '', ''),
    (14, 'P 1 21/c 1', '-P 2ybc', '', ''),
    (14, 'P 1 21/n 1', '-P 2yn', '', ''),
    (14, 'P 1 21/a 1', '-P 2yab', '', ''),
    (14, 'P 1 1 21/a', '-P 2ac', '', ''),
    (14, 'P 1 1 21/n', '-P 2n', '', ''),
    (14, 'P 1 1 21/b', '-P 2bc', '', ''),
    (14, 'P 21/b 1 1', '-P 2xab', '', ''),
    (14, 'P 21/n 1 1', '-P 2xn', '', ''),
    (14, 'P 21/c 1 1', '-P 2xac', '', ''),
    (15, 'C 1 2/c 1', '-C 2yc', '', ''),
    (15, 'A 1 2/n 1', '-A 2yab', '', ''),
    (15, 'I 1 2/a 1', '-I 2ya', '', ''),
    (15, 'A 1 2/a 1', '-A 2ya', '', ''),
    (15, 'C 1 2/n 1', '-C 2yac', '', ''),
    (15, 'I 1 2/c 1', '-I 2yc', '', ''),
    (15, 'A 1 1 2/a', '-A 2a', '', ''),
    (15, 'B 1 1 2/n', '-B 2ab', '', ''),
    (15, 'I 1 1 2/b', '-I 2b', '', ''),
    (15, 'B 1 1 2/b', '-B 2b', '', ''),
    (15, 'A 1 1 2/n', '-A 2ab', '', ''),
    (15, 'I 1 1 2/a', '-I 2a', '', ''),
    (15, 'B 2/b 1 1', '-B 2xb', '', ''),
    (15, 'C 2/n 1 1', '-C 2xac', '', ''),
    (15, 'I 2/c 1 1', '-I 2xc', '', ''),
    (15, 'C 2/c 1 1', '-C 2xc', '', ''),
    (15, 'B 2/n 1 1', '-B 2xab', '', ''),
    (15, 'I 2/b 1 1', '-I 2xb', '', ''),
    (16, 'P 2 2 2', 'P 2 2', '', ''),
    (17, 'P 2 2 21', 'P 2c 2', '', ''),
    (17, 'P 21 2 2', 'P 2a 2a', '', ''),
    (17, 'P 2 21 2', 'P 2 2b', '', ''),
    (18, 'P 21 21 2', 'P 2 2ab', '', ''),
    (18, 'P 2 21 21', 'P 2bc 2', '', ''),
    (18, 'P 21 2 21', 'P 2ac 2ac', '', ''),
    (19, 'P 21 21 21', 'P 2ac 2ab', '', ''),
    (20, 'C 2 2 21', 'C 2c 2', '', ''),
    (20, 'A 21 2 2', 'A 2a 2a', '', ''),
    (20, 'B 2 21 2', 'B 2 2b', '', ''),
    (21, 'C 2 2 2', 'C 2 2', '', ''),
    (21, 'A 2 2 2', 'A 2 2', '', ''),
    (21, 'B 2 2 2', 'B 2 2', '', ''),
    (22, 'F 2 2 2', 'F 2 2', '', ''),
    (23, 'I 2 2 2', 'I 2 2', '', ''),
    (24, 'I 21 21 21', 'I 2b 2c', '', ''),
    (25, 'P m m 2', 'P 2 -2', '', ''),
    (25, 'P 2 m m', 'P -2 2', '', ''),
    (25, 'P m 2 m', 'P -2 -2', '', ''),
    (26, 'P m c 21', 'P 2c -2', '', ''),
    (26, 'P c m 21', 'P 2c -2c', '', ''),
    (26, 'P 21 m a', 'P -2a 2a', '', ''),
    (26, 'P 21 a m', 'P -2 2a', '', ''),
    (26, 'P b 21 m', 'P -2 -2b', '', ''),
    (26, 'P m 21 b', 'P -2b -2', '', ''),
    (27, 'P c c 2', 'P 2 -2c', '', ''),
    (27, 'P 2 a a', 'P -2a 2', '', ''),
    (27, 'P b 2 b', 'P -2b -2b', '', ''),
    (28, 'P m a 2', 'P 2 -2a', '', ''),
    (28, 'P b m 2', 'P 2 -2b', '', ''),
    (28, 'P 2 m b', 'P -2b 2', '', ''),
    (28, 'P 2 c m', 'P -2c 2', '', ''),
    (28, 'P c 2 m', 'P -2c -2c', '', ''),
    (28, 'P m 2 a', 'P -2a -2a', '', ''),
    (29, 'P c a 21', 'P 2c -2ac', '', ''),
    (29, 'P b c 21', 'P 2c -2b', '', ''),
    (29, 'P 21 a b', 'P -2b 2a', '', ''),
    (29, 'P 21 c a', 'P -2ac 2a', '', ''),
    (29, 'P c 21 b', 'P -2bc -2c', '', ''),
    (29, 'P b 21 a', 'P -2a -2ab', '', ''),
    (30, 'P n c 2', 'P 2 -2bc', '', ''),
    (30, 'P c n 2', 'P 2 -2ac', '', ''),
    (30, 'P 2 n a', 'P -2ac 2', '', ''),
    (30, 'P 2 a n', 'P -2ab 2', '', ''),
    (30, 'P b 2 n', 'P -2ab -2ab', '', ''),
    (30, 'P n 2 b', 'P -2bc -2bc', '', ''),
    (31, 'P m n 21', 'P 2ac -2', '', ''),
    (31, 'P n m 21', 'P 2bc -2bc', '', ''),
    (31, 'P 21 m n', 'P -2ab 2ab', '', ''),
    (31, 'P 21 n m', 'P -2 2ac', '', ''),
    (31, 'P n 21 m', 'P -2 -2bc', '', ''),
    (31, 'P m 21 n', 'P -2ab -2', '', ''),
    (32, 'P b a 2', 'P 2 -2ab', '', ''),
    (32, 'P 2 c b', 'P -2bc 2', '', ''),
    (32, 'P c 2 a', 'P -2ac -2ac', '', ''),
    (33, 'P n a 21', 'P 2c -2n', '', ''),
    (33, 'P b n 21', 'P 2c -2ab', '', ''),
    (33, 'P 21 n b', 'P -2bc 2a', '', ''),
    (33, 'P 21 c n', 'P -2n 2a', '', ''),
    (33, 'P c 21 n', 'P -2n -2ac', '', ''),
    (33, 'P n 21 a', 'P -2ac -2n', '', ''),
    (34, 'P n n 2', 'P 2 -2n', '', ''),
    (34, 'P 2 n n', 'P -2n 2', '', ''),
    (34, 'P n 2 n', 'P -2n -2n', '', ''),
    (35, 'C m m 2', 'C 2 -2', '', ''),
    (35, 'A 2 m m', 'A -2 2', '', ''),
    (35, 'B m 2 m', 'B -2 -2', '', ''),
    (36, 'C m c 21', 'C 2c -2', '', ''),
    (36, 'C c m 21', 'C 2c -2c', '', ''),
    (36, 'A 21 m a', 'A -2a 2a', '', ''),
    (36, 'A 21 a m', 'A -2 2a', '', ''),
    (36, 'B b 21 m', 'B -2 -2b', '', ''),
    (36, 'B m 21 b', 'B -2b -2', '', ''),
    (37, 'C c c 2', 'C 2 -2c', '', ''),
    (37, 'A 2 a a', 'A -2a 2', '', ''),
    (37, 'B b 2 b', 'B -2b -2b', '', ''),
    (38, 'A m m 2', 'A 2 -2', '', ''),
    (38, 'B m m 2', 'B 2 -2', '', ''),
    (38, 'B 2 m m', 'B -2 2', '', ''),
    (38, 'C 2 m m', 'C -2 2', '', ''),
    (38, 'C m 2 m', 'C -2 -2', '', ''),
    (38, 'A m 2 m', 'A -2 -2', '', ''),
    (39, 'A b m 2', 'A 2 -2b', '', 'A e m 2'),
    (39, 'B m a 2', 'B 2 -2a', '', 'B m e 2'),
    (39, 'B 2 c m', 'B -2a 2', '', 'B 2 e m'),
    (39, 'C 2 m b', 'C -2a 2', '', 'C 2 m e'),
    (39, 'C m 2 a', 'C -2a -2a', '', 'C m 2 e'),
    (39, 'A c 2 m', 'A -2b -2b', '', 'A e 2 m'),
    (40, 'A m a 2', 'A 2 -2a', '', ''),
    (40, 'B b m 2', 'B 2 -2b', '', ''),
    (40, 'B 2 m b', 'B -2b 2', '', ''),
    (40, 'C 2 c m', 'C -2c 2', '', ''),
    (40, 'C c 2 m', 'C -2c -2c', '', ''),
    (40, 'A m 2 a', 'A -2a -2a', '', ''),
    (41, 'A b a 2', 'A 2 -2ab', '', 'A e a 2'),
    (41, 'B b a 2', 'B 2 -2ab', '', 'B b e 2'),
    (41, 'B 2 c b', 'B -2ab 2', '', 'B 2 e b'),
    (41, 'C 2 c b', 'C -2ac 2', '', 'C 2 c e'),
    (41, 'C c 2 a', 'C -2ac -2ac', '', 'C c 2 e'),
    (41, 'A c 2 a', 'A -2ab -2ab', '', 'A e 2 a'),
    (42, 'F m m 2', 'F 2 -2', '', ''),
    (42, 'F 2 m m', 'F -2 2', '', ''),
    (42, 'F m 2 m', 'F -2 -2', '', ''),
    (43, 'F d d 2', 'F 2 -2d', '', ''),
    (43, 'F 2 d d', 'F -2d 2', '', ''),
    (43, 'F d 2 d', 'F -2d -2d', '', ''),
    (44, 'I m m 2', 'I 2 -2', '', ''),
    (44, 'I 2 m m', 'I -2 2', '', ''),
    (44, 'I m 2 m', 'I -2 -2', '', ''),
    (45, 'I b a 2', 'I 2 -2c', '', ''),
    (45, 'I 2 c b', 'I -2a 2', '', ''),
    (45, 'I c 2 a', 'I -2b -2b', '', ''),
    (46, 'I m a 2', 'I 2 -2a', '', ''),
    (46, 'I b m 2', 'I 2 -2b', '', ''),
    (46, 'I 2 m b', 'I -2b 2', '', ''),
    (46, 'I 2 c m', 'I -2c 2', '', ''),
    (46, 'I c 2 m', 'I -2c -2c', '', ''),
    (46, 'I m 2 a', 'I -2a -2a', '', ''),
    (47, 'P m m m', '-P 2 2', '', ''),
    (48, 'P n n n', 'P 2 2 -1n', '1', ''),
    (48, 'P n n n', '-P 2ab 2bc', '2', ''),
    (49, 'P c c m', '-P 2 2c', '', ''),
    (49, 'P m a a', '-P 2a 2', '', ''),
    (49, 'P b m b', '-P 2b 2b', '', ''),
    (50, 'P b a n', 'P 2 2 -1ab', '1', ''),
    (50, 'P b a n', '-P 2ab 2b', '2', ''),
    (50, 'P n c b', 'P 2 2 -1bc', '1', ''),
    (50, 'P n c b', '-P 2b 2bc', '2', ''),
    (50, 'P c n a', 'P 2 2 -1ac', '1', ''),
    (50, 'P c n a', '-P 2a 2c', '2', ''),
    (51, 'P m m a', '-P 2a 2a', '', ''),
    (51, 'P m m b', '-P 2b 2', '', ''),
    (51, 'P b m m', '-P 2 2b', '', ''),
    (51, 'P c m m', '-P 2c 2c', '', ''),
    (51, 'P m c m', '-P 2c 2', '', ''),
    (51, 'P m a m', '-P 2 2a', '', ''),
    (52, 'P n n a', '-P 2a 2bc', '', ''),
    (52, 'P n n b', '-P 2b 2n', '', ''),
    (52, 'P b n n', '-P 2n 2b', '', ''),
    (52, 'P c n n', '-P 2ab 2c', '', ''),
    (52, 'P n c n', '-P 2ab 2n', '', ''),
    (52, 'P n a n', '-P 2n 2bc', '', ''),
    (53, 'P m n a', '-P 2ac 2', '', ''),
    (53, 'P n m b', '-P 2bc 2bc', '', ''),
    (53, 'P b m n', '-P 2ab 2ab', '', ''),
    (53, 'P c n m', '-P 2 2ac', '', ''),
    (53, 'P n c m', '-P 2 2bc', '', ''),
    (53, 'P m a n', '-P 2ab 2', '', ''),
    (54, 'P c c a', '-P 2a 2ac', '', ''),
    (54, 'P c c b', '-P 2b 2c', '', ''),
    (54, 'P b a a', '-P 2a 2b', '', ''),
    (54, 'P c a a', '-P 2ac 2c', '', ''),
    (54, 'P b c b', '-P 2bc 2b', '', ''),
    (54, 'P b a b', '-P 2b 2ab', '', ''),
    (55, 'P b a m', '-P 2 2ab', '', ''),
    (55, 'P m c b', '-P 2bc 2', '', ''),
    (55, 'P c m a', '-P 2ac 2ac', '', ''),
    (56, 'P c c n', '-P 2ab 2ac', '', ''),
    (56, 'P n a a', '-P 2ac 2bc', '', ''),
    (56, 'P b n b', '-P 2bc 2ab', '', ''),
    (57, 'P b c m', '-P 2c 2b', '', ''),
    (57, 'P c a m', '-P 2c 2ac', '', ''),
    (57, 'P m c a', '-P 2ac 2a', '', ''),
    (57, 'P m a b', '-P 2b 2a', '', ''),
    (57, 'P b m a', '-P 2a 2ab', '', ''),
    (57, 'P c m b', '-P 2bc 2c', '', ''),
    (58, 'P n n m', '-P 2 2n', '', ''),
    (58, 'P m n n', '-P 2n 2', '', ''),
    (58, 'P n m n', '-P 2n 2n', '', ''),
    (59, 'P m m n', 'P 2 2ab -1ab', '1', ''),
    (59, 'P m m n', '-P 2ab 2a', '2', ''),
    (59, 'P n m m', 'P 2bc 2 -1bc', '1', ''),
    (59, 'P n m m', '-P 2c 2bc', '2', ''),
    (59, 'P m n m', 'P 2ac 2ac -1ac', '1', ''),
    (59, 'P m n m', '-P 2c 2a', '2', ''),
    (60, 'P b c n', '-P 2n 2ab', '', ''),
    (60, 'P c a n', '-P 2n 2c', '', ''),
    (60, 'P n c a', '-P 2a 2n', '', ''),
    (60, 'P n a b', '-P 2bc 2n', '', ''),
    (60, 'P b n a', '-P 2ac 2b', '', ''),
    (60, 'P c n b', '-P 2b 2ac', '', ''),
    (61, 'P b c a', '-P 2ac 2ab', '', ''),
    (61, 'P c a b', '-P 2bc 2ac', '', ''),
    (62, 'P n m a', '-P 2ac 2n', '', ''),
    (62, 'P m n b', '-P 2bc 2a', '', ''),
    (62, 'P b n m', '-P 2c 2ab', '', ''),
    (62, 'P c m n', '-P 2n 2ac', '', ''),
    (62, 'P m c n', '-P 2n 2a', '', ''),
    (62, 'P n a m', '-P 2c 2n', '', ''),
    (63, 'C m c m', '-C 2c 2', '', ''),
    (63, 'C c m m', '-C 2c 2c', '', ''),
    (63, 'A m m a', '-A 2a 2a', '', ''),
    (63, 'A m a m', '-A 2 2a', '', ''),
    (63, 'B b m m', '-B 2 2b', '', ''),
    (63, 'B m m b', '-B 2b 2', '', ''),
    (64, 'C m c a', '-C 2ac 2', '', 'C m c e'),
    (64, 'C c m b', '-C 2ac 2ac', '', 'C c m e'),
    (64, 'A b m a', '-A 2ab 2ab', '', 'A e m a'),
    (64, 'A c a m', '-A 2 2ab', '', 'A e a m'),
    (64, 'B b c m', '-B 2 2ab', '', 'B b e m'),
    (64, 'B m a b', '-B 2ab 2', '', 'B m e b'),
    (65, 'C m m m', '-C 2 2', '', ''),
    (65, 'A m m m', '-A 2 2', '', ''),
    (65, 'B m m m', '-B 2 2', '', ''),
    (66, 'C c c m', '-C 2 2c', '', ''),
    (66, 'A m a a', '-A 2a 2', '', ''),
    (66, 'B b m b', '-B 2b 2b', '', ''),
    (67, 'C m m a', '-C 2a 2', '', 'C m m e'),
    (67, 'C m m b', '-C 2a 2a', '', 'C m m e'),
    (67, 'A b m m', '-A 2b 2b', '', 'A e m m'),
    (67, 'A c m m', '-A 2 2b', '', 'A e m m'),
    (67, 'B m c m', '-B 2 2a', '', 'B m e m'),
    (67, 'B m a m', '-B 2a 2', '', 'B m e m'),
    (68, 'C c c a', 'C 2 2 -1ac', '1', 'C c c e'),
    (68, 'C c c a', '-C 2a 2ac', '2', 'C c c e'),
    (68, 'C c c b', 'C 2 2 -1ac', '1', 'C c c e'),
    (68, 'C c c b', '-C 2a 2c', '2', 'C c c e'),
    (68, 'A b a a', 'A 2 2 -1ab', '1', 'A e a a'),
    (68, 'A b a a', '-A 2a 2b', '2', 'A e a a'),
    (68, 'A c a a', 'A 2 2 -1ab', '1', 'A e a a'),
    (68, 'A c a a', '-A 2ab 2b', '2', 'A e a a'),
    (68, 'B b c b', 'B 2 2 -1ab', '1', 'B b e b'),
    (68, 'B b c b', '-B 2ab 2b', '2', 'B b e b'),
    (68, 'B b a b', 'B 2 2 -1ab', '1', 'B b e b'),
    (68, 'B b a b', '-B 2b 2ab', '2', 'B b e b'),
    (69, 'F m m m', '-F 2 2', '', ''),
    (70, 'F d d d', 'F 2 2 -1d', '1', ''),
    (70, 'F d d d', '-F 2uv 2vw', '2', ''),
    (71, 'I m m m', '-I 2 2', '', ''),
    (72, 'I b a m', '-I 2 2c', '', ''),
    (72, 'I m c b', '-I 2a 2', '', ''),
    (72, 'I c m a', '-I 2b 2b', '', ''),
    (73, 'I b c a', '-I 2b 2c', '', ''),
    (73, 'I c a b', '-I 2a 2b', '', ''),
    (74, 'I m m a', '-I 2b 2', '', ''),
    (74, 'I m m b', '-I 2a 2a', '', ''),
    (74, 'I b m m', '-I 2c 2c', '', ''),
    (74, 'I c m m', '-I 2 2b', '', ''),
    (74, 'I m c m', '-I 2 2a', '', ''),
    (74, 'I m a m', '-I 2c 2', '', ''),
    (75, 'P 4', 'P 4', '', ''),
    (76, 'P 41', 'P 4w', '', ''),
    (77, 'P 42', 'P 4c', '', ''),
    (78, 'P 43', 'P 4cw', '', ''),
    (79, 'I 4', 'I 4', '', ''),
    (80, 'I 41', 'I 4bw', '', ''),
    (81, 'P -4', 'P -4', '', ''),
    (82, 'I -4', 'I -4', '', ''),
    (83, 'P 4/m', '-P 4', '', ''),
    (84, 'P 42/m', '-P 4c', '', ''),
    (85, 'P 4/n', 'P 4ab -1ab', '1', ''),
    (85, 'P 4/n', '-P 4a', '2', ''),
    (86, 'P 42/n', 'P 4n -1n', '1', ''),
    (86, 'P 42/n', '-P 4bc', '2', ''),
    (87, 'I 4/m', '-I 4', '', ''),
    (88, 'I 41/a', 'I 4bw -1bw', '1', ''),
    (88, 'I 41/a', '-I 4ad', '2', ''),
    (89, 'P 4 2 2', 'P 4 2', '', ''),
    (90, 'P 4 21 2', 'P 4ab 2ab', '', ''),
    (91, 'P 41 2 2', 'P 4w 2c', '', ''),
    (92, 'P 41 21 2', 'P 4abw 2nw', '', ''),
    (93, 'P 42 2 2', 'P 4c 2', '', ''),
    (94, 'P 42 21 2', 'P 4n 2n', '', ''),
    (95, 'P 43 2 2', 'P 4cw 2c', '', ''),
    (96, 'P 43 21 2', 'P 4nw 2abw', '', ''),
    (97, 'I 4 2 2', 'I 4 2', '', ''),
    (98, 'I 41 2 2', 'I 4bw 2bw', '', ''),
    (99, 'P 4 m m', 'P 4 -2', '', ''),
    (100, 'P 4 b m', 'P 4 -2ab', '', ''),
    (101, 'P 42 c m', 'P 4c -2c', '', ''),
    (102, 'P 42 n m', 'P 4n -2n', '', ''),
    (103, 'P 4 c c', 'P 4 -2c', '', ''),
    (104, 'P 4 n c', 'P 4 -2n', '', ''),
    (105, 'P 42 m c', 'P 4c -2', '', ''),
    (106, 'P 42 b c', 'P 4c -2ab', '', ''),
    (107, 'I 4 m m', 'I 4 -2', '', ''),
    (108, 'I 4 c m', 'I 4 -2c', '', ''),
    (109, 'I 41 m d', 'I 4bw -2', '', ''),
    (110, 'I 41 c d', 'I 4bw -2c', '', ''),
    (111, 'P -4 2 m', 'P -4 2', '', ''),
    (112, 'P -4 2 c', 'P -4 2c', '', ''),
    (113, 'P -4 21 m', 'P -4 2ab', '', ''),
    (114, 'P -4 21 c', 'P -4 2n', '', ''),
    (115, 'P -4 m 2', 'P -4 -2', '', ''),
    (116, 'P -4 c 2', 'P -4 -2c', '', ''),
    (117, 'P -4 b 2', 'P -4 -2ab', '', ''),
    (118, 'P -4 n 2', 'P -4 -2n', '', ''),
    (119, 'I -4 m 2', 'I -4 -2', '', ''),
    (120, 'I -4 c 2', 'I -4 -2c', '', ''),
    (121, 'I -4 2 m', 'I -4 2', '', ''),
    (122, 'I -4 2 d', 'I -4 2bw', '', ''),
    (123, 'P 4/m m m', '-P 4 2', '', ''),
    (124, 'P 4/m c c', '-P 4 2c', '', ''),
    (125, 'P 4/n b m', 'P 4 2 -1ab', '1', ''),
    (125, 'P 4/n b m', '-P 4a 2b', '2', ''),
    (126, 'P 4/n n c', 'P 4 2 -1n', '1', ''),
    (126, 'P 4/n n c', '-P 4a 2bc', '2', ''),
    (127, 'P 4/m b m', '-P 4 2ab', '', ''),
    (128, 'P 4/m n c', '-P 4 2n', '', ''),
    (129, 'P 4/n m m', 'P 4ab 2ab -1ab', '1', ''),
    (129, 'P 4/n m m', '-P 4a 2a', '2', ''),
    (130, 'P 4/n c c', 'P 4ab 2n -1ab', '1', ''),
    (130, 'P 4/n c c', '-P 4a 2ac', '2', ''),
    (131, 'P 42/m m c', '-P 4c 2', '', ''),
    (132, 'P 42/m c m', '-P 4c 2c', '', ''),
    (133, 'P 42/n b c', 'P 4n 2c -1n', '1', ''),
    (133, 'P 42/n b c', '-P 4ac 2b', '2', ''),
    (134, 'P 42/n n m', 'P 4n 2 -1n', '1', ''),
    (134, 'P 42/n n m', '-P 4ac 2bc', '2', ''),
    (135, 'P 42/m b c', '-P 4c 2ab', '', ''),
    (136, 'P 42/m n m', '-P 4n 2n', '', ''),
    (137, 'P 42/n m c', 'P 4n 2n -1n', '1', ''),
    (137, 'P 42/n m c', '-P 4ac 2a', '2', ''),
    (138, 'P 42/n c m', 'P 4n 2ab -1n', '1', ''),
    (138, 'P 42/n c m', '-P 4ac 2ac', '2', ''),
    (139, 'I 4/m m m', '-I 4 2', '', ''),
    (140, 'I 4/m c m', '-I 4 2c', '', ''),
    (141, 'I 41/a m d', 'I 4bw 2bw -1bw', '1', ''),
    (141, 'I 41/a m d', '-I 4bd 2', '2', ''),
    (142, 'I 41/a c d', 'I 4bw 2aw -1bw', '1', ''),
    (142, 'I 41/a c d', '-I 4bd 2c', '2', ''),
    (143, 'P 3', 'P 3', '', ''),
    (144, 'P 31', 'P 31', '', ''),
    (145, 'P 32', 'P 32', '', ''),
    (146, 'R 3', 'R 3', 'H', ''),
    (146, 'R 3', 'P 3*', 'R', ''),
    (147, 'P -3', '-P 3', '', ''),
    (148, 'R -3', '-R 3', 'H', ''),
    (148, 'R -3', '-P 3*', 'R', ''),
    (149, 'P 3 1 2', 'P 3 2', '', ''),
    (150, 'P 3 2 1', 'P 3 2"', '', ''),
    (151, 'P 31 1 2', 'P 31 2 (0 0 4)', '', ''),
    (152, 'P 31 2 1', 'P 31 2"', '', ''),
    (153, 'P 32 1 2', 'P 32 2 (0 0 2)', '', ''),
    (154, 'P 32 2 1', 'P 32 2"', '', ''),
    (155, 'R 3 2', 'R 3 2"', 'H', ''),
    (155, 'R 3 2', 'P 3* 2', 'R', ''),
    (156, 'P 3 m 1', 'P 3 -2"', '', ''),
    (157, 'P 3 1 m', 'P 3 -2', '', ''),
    (158, 'P 3 c 1', 'P 3 -2"c', '', ''),
    (159, 'P 3 1 c', 'P 3 -2c', '', ''),
    (160, 'R 3 m', 'R 3 -2"', 'H', ''),
    (160, 'R 3 m', 'P 3* -2', 'R', ''),
    (161, 'R 3 c', 'R 3 -2"c', 'H', ''),
    (161, 'R 3 c', 'P 3* -2n', 'R', ''),
    (162, 'P -3 1 m', '-P 3 2', '', ''),
    (163, 'P -3 1 c', '-P 3 2c', '', ''),
    (164, 'P -3 m 1', '-P 3 2"', '', ''),
    (165, 'P -3 c 1', '-P 3 2"c', '', ''),
    (166, 'R -3 m', '-R 3 2"', 'H', ''),
    (166, 'R -3 m', '-P 3* 2', 'R', ''),
    (167, 'R -3 c', '-R 3 2"c', 'H', ''),
    (167, 'R -3 c', '-P 3* 2n', 'R', ''),
    (168, 'P 6', 'P 6', '', ''),
    (169, 'P 61', 'P 61', '', ''),
    (170, 'P 65', 'P 65', '', ''),
    (171, 'P 62', 'P 62', '', ''),
    (172, 'P 64', 'P 64', '', ''),
    (173, 'P 63', 'P 6c', '', ''),
    (174, 'P -6', 'P -6', '', ''),
    (175, 'P 6/m', '-P 6', '', ''),
    (176, 'P 63/m', '-P 6c', '', ''),
    (177, 'P 6 2 2', 'P 6 2', '', ''),
    (178, 'P 61 2 2', 'P 61 2 (0 0 5)', '', ''),
    (179, 'P 65 2 2', 'P 65 2 (0 0 1)', '', ''),
    (180, 'P 62 2 2', 'P 62 2 (0 0 4)', '', ''),
    (181, 'P 64 2 2', 'P 64 2 (0 0 2)', '', ''),
    (182, 'P 63 2 2', 'P 6c 2c', '', ''),
    (183, 'P 6 m m', 'P 6 -2', '', ''),
    (184, 'P 6 c c', 'P 6 -2c', '', ''),
    (185, 'P 63 c m', 'P 6c -2', '', ''),
    (186, 'P 63 m c', 'P 6c -2c', '', ''),
    (187, 'P -6 m 2', 'P -6 2', '', ''),
    (188, 'P -6 c 2', 'P -6c 2', '', ''),
    (189, 'P -6 2 m', 'P -6 -2', '', ''),
    (190, 'P -6 2 c', 'P -6c -2c', '', ''),
    (191, 'P 6/m m m', '-P 6 2', '', ''),
    (192, 'P 6/m c c', '-P 6 2c', '', ''),
    (193, 'P 63/m c m', '-P 6c 2', '', ''),
    (194, 'P 63/m m c', '-P 6c 2c', '', ''),
    (195, 'P 2 3', 'P 2 2 3', '', ''),
    (196, 'F 2 3', 'F 2 2 3', '', ''),
    (197, 'I 2 3', 'I 2 2 3', '', ''),
    (198, 'P 21 3', 'P 2ac 2ab 3', '', ''),
    (199, 'I 21 3', 'I 2b 2c 3', '', ''),
    (200, 'P m -3', '-P 2 2 3', '', ''),
    (201, 'P n -3', 'P 2 2 3 -1n', '1', ''),
    (201, 'P n -3', '-P 2ab 2bc 3', '2', ''),
    (202, 'F m -3', '-F 2 2 3', '', ''),
    (203, 'F d -3', 'F 2 2 3 -1d', '1', ''),
    (203, 'F d -3', '-F 2uv 2vw 3', '2', ''),
    (204, 'I m -3', '-I 2 2 3', '', ''),
    (205, 'P a -3', '-P 2ac 2ab 3', '', ''),
    (206, 'I a -3', '-I 2b 2c 3', '', ''),
    (207, 'P 4 3 2', 'P 4 2 3', '', ''),
    (208, 'P 42 3 2', 'P 4n 2 3', '', ''),
    (209, 'F 4 3 2', 'F 4 2 3', '', ''),
    (210, 'F 41 3 2', 'F 4d 2 3', '', ''),
    (211, 'I 4 3 2', 'I 4 2 3', '', ''),
    (212, 'P 43 3 2', 'P 4acd 2ab 3', '', ''),
    (213, 'P 41 3 2', 'P 4bd 2ab 3', '', ''),
    (214, 'I 41 3 2', 'I 4bd 2c 3', '', ''),
    (215, 'P -4 3 m', 'P -4 2 3', '', ''),
    (216, 'F -4 3 m', 'F -4 2 3', '', ''),
    (217, 'I -4 3 m', 'I -4 2 3', '', ''),
    (218, 'P -4 3 n', 'P -4n 2 3', '', ''),
    (219, 'F -4 3 c', 'F -4a 2 3', '', ''),
    (220, 'I -4 3 d', 'I -4bd 2c 3', '', ''),
    (221, 'P m -3 m', '-P 4 2 3', '', ''),
    (222, 'P n -3 n', 'P 4 2 3 -1n', '1', ''),
    (222, 'P n -3 n', '-P 4a 2bc 3', '2', ''),
    (223, 'P m -3 n', '-P 4n 2 3', '', ''),
    (224, 'P n -3 m', 'P 4n 2 3 -1n', '1', ''),
    (224, 'P n -3 m', '-P 4bc 2bc 3', '2', ''),
    (225, 'F m -3 m', '-F 4 2 3', '', ''),
    (226, 'F m -3 c', '-F 4a 2 3', '', ''),
    (227, 'F d -3 m', 'F 4d 2 3 -1d', '1', ''),
    (227, 'F d -3 m', '-F 4vw 2vw 3', '2', ''),
    (228, 'F d -3 c', 'F 4d 2 3 -1ad', '1', ''),
    (228, 'F d -3 c', '-F 4ud 2vw 3', '2', ''),
    (229, 'I m -3 m', '-I 4 2 3', '', ''),
    (230, 'I a -3 d', '-I 4bd 2c 3', '', ''),
    (5, 'I 1 21 1', 'I 2yb', '', ''),
    (5, 'C 1 21 1', 'C 2yb', '', ''),
    (18, 'P 21212(a)', 'P 2ab 2a', '', ''),
    (20, 'C 2 2 21a)', 'C 2ac 2', '', ''),
    (21, 'C 2 2 2a', 'C 2ab 2b', '', ''),
    (22, 'F 2 2 2a', 'F 2 2c', '', ''),
    (23, 'I 2 2 2a', 'I 2ab 2bc', '', ''),
    (94, 'P 42 21 2a', 'P 4bc 2a', '', ''),
    (197, 'I 2 3a', 'I 2ab 2bc 3', '', ''),
    (1, 'A 1', 'A 1', '', ''),
    (1, 'B 1', 'B 1', '', ''),
    (1, 'C 1', 'C 1', '', ''),
    (1, 'F 1', 'F 1', '', ''),
    (1, 'I 1', 'I 1', '', ''),
    (2, 'A -1', '-A 1', '', ''),
    (2, 'B -1', '-B 1', '', ''),
    (2, 'C -1', '-C 1', '', ''),
    (2, 'F -1', '-F 1', '', ''),
    (2, 'I -1', '-I 1', '', ''),
    (3, 'B 1 2 1', 'B 2y', '', ''),
    (3, 'C 1 1 2', 'C 2', '', ''),
    (4, 'B 1 21 1', 'B 2yb', '', ''),
    (4, 'C 1 1 21', 'C 2c', '', ''),
    (5, 'F 1 2 1', 'F 2y', '', ''),
    (8, 'F 1 m 1', 'F -2y', '', ''),
    (9, 'F 1 d 1', 'F -2yuw', '', ''),
    (12, 'F 1 2/m 1', '-F 2y', '', ''),
    (64, 'A b a m', '-A 2 2ab', '', 'A e a m'),
    (89, 'C 4 2 2', 'C 4 2', '', ''),
    (90, 'C 4 2 21', 'C 4a 2', '', ''),
    (97, 'F 4 2 2', 'F 4 2', '', ''),
    (115, 'C -4 2 m', 'C -4 2', '', ''),
    (117, 'C -4 2 b', 'C -4 2ya', '', ''),
    (139, 'F 4/m m m', '-F 4 2', '', ''),
)


def _norm(s: str) -> str:
    return s.replace(" ", "").replace("_", "").lower()


def display_hm(hm: str, tag: str, hm_2016: str = "") -> str:
    """ITA name of one setting, with origin choice or rhombohedral axes spelled out."""
    base = hm_2016 or hm
    if tag == "2":
        return f"{base} (origin choice 2)"
    if tag == "R":
        return f"{base} (rhombohedral axes)"
    return base


def _index():
    """symbol → row. A bare name is registered only when it is unambiguous."""
    by_norm: dict[str, list] = {}
    by_hall: dict[str, tuple] = {}
    for row in ITA_SETTINGS:
        hm, hall, tag, hm_2016 = row[1], row[2], row[3], row[4]
        by_hall[hall] = row
        keys = {_norm(hm), _norm(display_hm(hm, tag, hm_2016))}
        if hm_2016:
            keys.add(_norm(hm_2016))
            keys.add(_norm(display_hm(hm, tag, hm_2016)))
        if tag in ("1", "2", "H", "R"):
            keys.add(_norm(f"{hm}:{tag}"))
            if hm_2016:
                keys.add(_norm(f"{hm_2016}:{tag}"))
        for key in keys:
            by_norm.setdefault(key, []).append(row)
    unique = {}
    for key, hits in by_norm.items():
        # Same Hall listed twice is one setting.
        halls = {h[2] for h in hits}
        if len(halls) == 1:
            unique[key] = hits[0]
    return unique, by_hall


_BY_NORM, _BY_HALL = _index()


# Monoclinic numbers in ITA Table 4.3.2.1. Short names and ``14:b2`` codes
# are derived from these rows; nothing is typed by hand.
_MONOCLINIC = range(3, 16)
_SCREW_PAREN = re.compile(r"\((\d)\)")


def _strip_screw_parens(text: str) -> str:
    """Turn ``P2(1)/c`` into ``P21/c``. A change of basis has commas and is left alone."""
    if "," in text:
        return text
    return _SCREW_PAREN.sub(r"\1", text)


def _unique_axis_of(hm: str) -> str | None:
    """``'b'`` for ``P 1 21/n 1``, ``'c'`` for ``P 1 1 21/a``, ``'a'`` for ``P 21/c 1 1``."""
    parts = hm.split()
    if len(parts) != 4:
        return None
    found = ["abc"[i] for i, tok in enumerate(parts[1:]) if tok != "1"]
    if len(found) == 1:
        return found[0]
    return None


def _compress_monoclinic(hm: str) -> str | None:
    """Drop explicit ``1`` axis tokens: ``P 1 21/n 1`` → ``P21/n``."""
    parts = hm.split()
    if _unique_axis_of(hm) is None:
        return None
    kept = [parts[0]] + [tok for tok in parts[1:] if tok != "1"]
    if len(kept) < 2:
        return None
    return "".join(kept)


def _build_aliases():
    """Short monoclinic names and ``number:setting`` codes, from table order."""
    short_hits: dict[str, list] = {}
    for row in ITA_SETTINGS:
        if row[0] not in _MONOCLINIC:
            continue
        axis = _unique_axis_of(row[1])
        short = _compress_monoclinic(row[1])
        if axis is None or short is None:
            continue
        short_hits.setdefault(_norm(short), []).append((row, axis))
    short_unique: dict[str, tuple] = {}
    for key, hits in short_hits.items():
        halls = {hit[0][2] for hit in hits}
        if len(halls) == 1:
            short_unique[key] = (hits[0][0], None)
            continue
        b_hits = [hit for hit in hits if hit[1] == "b"]
        b_halls = {hit[0][2] for hit in b_hits}
        if len(b_halls) == 1:
            short_unique[key] = (b_hits[0][0], "unique axis b")
    codes: dict[str, tuple] = {}
    by_axis: dict[tuple, list] = {}
    for row in ITA_SETTINGS:
        if row[0] not in _MONOCLINIC:
            continue
        axis = _unique_axis_of(row[1])
        if axis is None:
            continue
        by_axis.setdefault((row[0], axis), []).append(row)
    for (number, axis), rows in by_axis.items():
        for index, row in enumerate(rows, start=1):
            codes[_norm(f"{number}:{axis}{index}")] = row
    for row in ITA_SETTINGS:
        tag = row[3]
        if tag in ("1", "2", "H", "R"):
            codes.setdefault(_norm(f"{row[0]}:{tag}"), row)
    return short_unique, codes


_SHORT, _CODE = _build_aliases()


def canonical_lookup_key(text: str) -> str:
    """Hall symbol when ``text`` is a short name, screw parenthesis, or ``number:setting``.

    Otherwise the screw-stripped text, which is ``text`` itself when nothing changed.
    """
    if not isinstance(text, str):
        return text
    stripped = _strip_screw_parens(text.strip())
    key = _norm(stripped)
    if key in _CODE:
        return _CODE[key][2]
    if key in _SHORT:
        return _SHORT[key][0][2]
    return stripped


def symbol_resolution(text: str) -> dict | None:
    """Alias metadata for a symbol, or None when ``text`` is already canonical.

    ``assumed`` is set only when a short name collapsed unique axes and the
    unique-axis-b row was chosen, which is the ITA default.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip()
    stripped = _strip_screw_parens(raw)
    key = _norm(stripped)
    if key in _CODE:
        return {"resolved_from": raw, "hall": _CODE[key][2]}
    if key in _SHORT:
        row, assumed = _SHORT[key]
        out = {"resolved_from": raw, "hall": row[2]}
        if assumed:
            out["assumed"] = assumed
        return out
    if stripped != raw:
        return {"resolved_from": raw}
    return None


def hm_short(row) -> str | None:
    """Compressed monoclinic symbol when parsing it returns this same Hall row.

    ``None`` when the row is not monoclinic, or the short name belongs to a
    different setting (the unique-axis-b default).
    """
    if row[0] not in _MONOCLINIC:
        return None
    short = _compress_monoclinic(row[1])
    if short is None:
        return None
    hit = _SHORT.get(_norm(short))
    if hit is None or hit[0][2] != row[2]:
        return None
    return short


def lookup_setting(key: str):
    """Return an ITA setting row for an extended symbol, or None.

    Accepts a Hermann–Mauguin symbol, its 2016 e-glide spelling, a ``:1``,
    ``:2``, ``:H`` or ``:R`` qualifier, a Hall symbol, a short monoclinic
    name (``P21/n``), a parenthesised screw (``P2(1)/c``), and an ITA
    ``number:setting`` code (``14:b2``, ``68:1``).
    """
    if not isinstance(key, str):
        return None
    text = canonical_lookup_key(key)
    if text in _BY_HALL:
        return _BY_HALL[text]
    return _BY_NORM.get(_norm(text))


def settings_for_number(number: int):
    """Every ITA setting row of one space-group number."""
    return [row for row in ITA_SETTINGS if row[0] == number]


def hall_for_ops(symbol: str):
    """Hall of the tabulated setting whose operations equal ``symbol``.

    ``symbol`` is parsed with the Hall parser. Returns None when it is not a
    Hall symbol or when no setting has the same operation set.
    """
    from .hall import ops_from_hall, parse_hall
    try:
        parse_hall(symbol)
        ops = ops_from_hall(symbol)
    except ValueError:
        return None
    for row in ITA_SETTINGS:
        if ops_from_hall(row[2]) == ops:
            return row[2]
    return None


def match_ops(number: int, ops):
    """The setting row of ``number`` whose operations are ``ops``, or None."""
    from .hall import ops_from_hall
    for row in settings_for_number(number):
        if ops_from_hall(row[2]) == ops:
            return row
    return None
