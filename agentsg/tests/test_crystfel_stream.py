"""Tests for the CrystFEL stream parser (unit-level, synthetic fixture)."""
import textwrap
import pytest
from agentsg.cell.crystfel_stream import parse_stream, read_cells, stream_summary

FIXTURE = textwrap.dedent("""\
CrystFEL stream format 2.3
----- Begin chunk -----
Image filename: img0.cxi
--- Begin crystal
Cell parameters 4.21213 4.21571 23.22923 nm, 90.00212 90.23685 120.01683 deg
astar = +0.2206153 +0.1324274 -0.0947101 nm^-1
bstar = +0.2345840 -0.1357014 -0.0400593 nm^-1
cstar = -0.0118513 -0.0087577 -0.0404491 nm^-1
lattice_type = hexagonal
--- End crystal
----- End chunk -----
----- Begin chunk -----
Image filename: img1.cxi
--- Begin crystal
Cell parameters 4.18000 4.19000 23.30000 nm, 90.0 90.0 120.0 deg
astar = +0.1 +0.2 +0.3 nm^-1
bstar = +0.4 +0.5 +0.6 nm^-1
cstar = +0.7 +0.8 +0.9 nm^-1
--- End crystal
----- End chunk -----
----- Begin chunk -----
Image filename: img2.cxi
----- End chunk -----
""")


@pytest.fixture
def stream_file(tmp_path):
    p = tmp_path / "test.stream"
    p.write_text(FIXTURE)
    return str(p)


def test_read_cells_count_and_units(stream_file):
    cells = read_cells(stream_file)
    assert len(cells) == 2                     # 2 indexed, 1 blank chunk
    a, b, c, al, be, ga = cells[0]
    assert abs(a - 42.1213) < 1e-3             # nm -> Angstrom
    assert abs(c - 232.2923) < 1e-3
    assert abs(ga - 120.01683) < 1e-3


def test_parse_stream_with_orientation(stream_file):
    xs = list(parse_stream(stream_file, with_orientation=True))
    assert len(xs) == 2
    assert xs[0]["astar"] == [0.2206153, 0.1324274, -0.0947101]
    assert xs[1]["cstar"] == [0.7, 0.8, 0.9]
    assert "cell" in xs[0]


def test_stream_summary(stream_file):
    s = stream_summary(stream_file)
    assert s["n_frames"] == 3
    assert s["n_indexed"] == 2
    assert abs(s["indexing_rate"] - 2 / 3) < 1e-9
    assert abs(s["cell_mean"][0] - (42.1213 + 41.8) / 2) < 1e-3


def test_root_invariant_of_parsed_cells(stream_file):
    """Parsed cells feed straight into the root-invariant machinery."""
    from agentsg.cell.rootform import root_invariant
    for cell in read_cells(stream_file):
        ri = root_invariant(cell)
        assert len(ri) == 6
