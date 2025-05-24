import os

import BioSimSpace.Sandpit.Exscientia as BSS  # type: ignore[import]
import pytest

from binding_affinity_predicting.simulation.parameterise import (
    _parameterise_water,
    parameterise_system,
)


@pytest.fixture(autouse=True)
def stub_parameterise_and_save(monkeypatch):
    """
    Stub out BSS.Parameters.parameterise and BSS.IO.saveMolecules so that
    parameterise_system will use real readMolecules but a fake parameterisation
    and write a dummy file for us to assert on.
    """

    # Fake parameterise: just return the same molecule in a trivial wrapper
    class FakeProcess:
        def __init__(self, mol, **kwargs):
            self._mol = mol

        def getMolecule(self):
            return self._mol

    def fake_parameterise(*, molecule, forcefield=None, water_model=None, **kw):
        return FakeProcess(molecule)

    monkeypatch.setattr(BSS.Parameters, "parameterise", fake_parameterise)

    # Fake saveMolecules: write a small marker file so our test can see it
    def fake_save(path, system, fileformat, **kw):
        """
        Write dummy .gro and .top files based on the fileformat list.
        e.g. path=".../combo", fileformat=["gro87","grotop"]
        -> write "combo.gro" and "combo.top"
        """
        ext_map = {"gro87": ".gro", "grotop": ".top"}
        for fmt in fileformat:
            ext = ext_map.get(fmt)
            if not ext:
                raise ValueError(f"Unsupported file format: {fmt}")
            out_fn = path + ext
            # make sure directory exists
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            with open(out_fn, "wb") as f:
                f.write(b"DUMMY")

    monkeypatch.setattr(BSS.IO, "saveMolecules", fake_save)


def test_parameterise_system_real_io(example_structures, out_dir):
    prot = str(example_structures / "protein.pdb")
    lig = str(example_structures / "ligand.sdf")

    parameterise_system(
        protein_path=prot,
        ligand_path=lig,
        filename_stem="combo",
        output_dir=str(out_dir),
    )

    # Now assert the GROMACS files are there and non‐empty:
    gro = out_dir / "combo.gro"
    top = out_dir / "combo.top"
    assert gro.exists() and gro.stat().st_size > 0, f"{gro} was not created or is empty"
    assert top.exists() and top.stat().st_size > 0, f"{top} was not created or is empty"


def test_parameterise_system_raises_if_no_paths():
    with pytest.raises(ValueError):
        parameterise_system()


def test_parameterise_water_only_io(example_structures, out_dir):
    wat = str(example_structures / "water.pdb")
    _parameterise_water(file_path=wat, filename_stem="water", output_dir=str(out_dir))
    # Now assert the GROMACS files are there and non‐empty:
    gro = out_dir / "water.gro"
    top = out_dir / "water.top"
    assert gro.exists() and gro.stat().st_size > 0, f"{gro} was not created or is empty"
    assert top.exists() and top.stat().st_size > 0, f"{top} was not created or is empty"
