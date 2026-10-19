"""COD explorer: linear-key plane, and no UniProt columns."""
from pathlib import Path

import numpy as np

from agentsg.cell import cod_explorer
from agentsg.cell.celldb import CellDatabase
from agentsg.cell.cod_explorer import PAGE, Archive, isomap, primitive_volume_scale, system_codes


_TABLE = """file,a,b,c,alpha,beta,gamma,sg,formula
1000001,5.43,5.43,5.43,90,90,90,F d -3 m,- Si8 -
1000002,5,6,7,90,90,90,P 1,- C -
1000003,8,9,10,90,90,90,P 21 21 21,- C4 -
"""


def test_primitive_volume_scale_matches_centring():
    scale = primitive_volume_scale(
        [8.0, 32.0, 8.0, 24.0],
        ["P 1", "F 2 3", "R 3 m :R", "R 3 m :H"],
    )
    assert np.allclose(scale, [2.0, 2.0, 2.0, 2.0])


def test_similar_cells_share_a_normalised_key(tmp_path):
    src = tmp_path / "cod.csv"
    src.write_text(
        "file,a,b,c,alpha,beta,gamma,sg,formula\n"
        "1000001,5,5,5,90,90,90,P 1,- A -\n"
        "1000002,10,10,10,90,90,90,P 1,- A -\n"
        "1000003,5,6,7,90,90,90,P 1,- B -\n"
    )
    db_path = tmp_path / "cod.duckdb"
    db = CellDatabase(db_path, archive="cod")
    db.add_cod(str(src))
    db.close()
    archive = Archive(db_path)
    try:
        same = archive.keys[archive.index["COD1000001"]] - archive.keys[archive.index["COD1000002"]]
        other = archive.keys[archive.index["COD1000001"]] - archive.keys[archive.index["COD1000003"]]
    finally:
        archive.close()
    assert np.linalg.norm(same) < 1e-5
    assert np.linalg.norm(other) > 0.05


def test_missing_space_group_stays_unknown():
    codes = system_codes(np.array([1, 0, 227]))
    assert codes.tolist() == [0, 7, 6]


def test_archive_card_has_formula_and_no_uniprot(tmp_path):
    src = tmp_path / "cod.csv"
    src.write_text(_TABLE)
    db_path = tmp_path / "cod.duckdb"
    db = CellDatabase(db_path, archive="cod")
    assert db.add_cod(str(src)) == (3, 0)
    db.close()

    archive = Archive(db_path)
    try:
        assert archive.xyz.shape == (3, 3)
        assert len(archive.variance) == 3
        assert archive.variance[0] >= archive.variance[1] >= 0
        assert archive.knn_idx.shape == (3, 16)
        embedded = isomap([0, 1, 2], archive.knn_idx, archive.knn_dist, archive.node_of_row)
        assert embedded["n_cells"] == 3 and embedded["n_lattices"] == 3
        card = archive.cell_card(archive.index["COD1000001"])
    finally:
        archive.close()

    assert card["formula"] == "- Si8 -"
    assert card["system"] == "cubic"
    assert card["url"] == "https://www.crystallography.net/cod/1000001.html"
    assert card["selling"]["a"]
    assert card["neighbors"]
    assert card["neighbors"][0]["id"] != "COD1000001"
    assert "uniprot" not in card
    assert "deposit_group" not in card
    assert "uniprot" not in PAGE.read_text().lower()
    source = Path(cod_explorer.__file__).read_text()
    assert "uniprots" not in source
    assert "deposit_group" not in source
