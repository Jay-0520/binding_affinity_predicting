import logging
import os
from typing import Optional

import BioSimSpace as BSS  # type: ignore[import]

from binding_affinity_predicting.simulation.utils import (
    rename_lig,
    save_system_to_local,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parameterise_system(
    *,
    protein_path: Optional[str] = None,
    ligand_path: Optional[str] = None,
    water_path: Optional[str] = None,
    protein_ff: str = "ff14SB",
    ligand_ff: str = "openff_unconstrained-2.0.0",
    water_ff: str = "ff14SB",
    water_model: str = "tip3p",
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> BSS._SireWrappers._system.System:  # type: ignore
    """
    Parameterise and assemble a combined System of any subset of protein, ligand and water.
    must pass at least one of `protein_path`, `ligand_path` or `water_path`.

    Parameters
    ----------
    protein_path: str
        PDB file for the protein (e.g. "protein.pdb").
    ligand_path: str
        SDF (or other) file for the ligand (e.g. "ligand.sdf").
    water_path: str
        PDB file for crystal waters (e.g. "water.pdb").
    protein_ff: str
        Force field for the protein (default "ff14SB").
    ligand_ff: str
        Force field for the ligand (default "openff_unconstrained-2.0.0").
    water_ff: str
        Force field for waters (default "ff14SB").
    water_model: str
        Water model identifier (default "tip3p").
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.

    Returns
    -------
    System
        The assembled, parameterised BioSimSpace System.
    """
    components = []

    if protein_path:
        prot_sys = _parameterise_protein(
            file_path=protein_path,
            forcefield=protein_ff,
        )
        components.append(prot_sys)

    if ligand_path:
        lig_sys = _parameterise_ligand(
            file_path=ligand_path,
            forcefield=ligand_ff,
        )
        components.append(lig_sys)

    if water_path:
        wat_sys = _parameterise_water(
            file_path=water_path,
            forcefield=water_ff,
            water_model=water_model,
        )
        components.append(wat_sys)

    if not components:
        raise ValueError(
            "Must supply at least one of protein_path, ligand_path or water_path!"
        )

    # Assemble into one System
    full_system = components[0]
    for sys_part in components[1:]:
        full_system += sys_part  # only works for different molecules

    # Save the system to local files if output_dir and filename_stem are provided
    save_system_to_local(
        system=full_system,
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    return full_system


def _parameterise_water(
    file_path: str,
    forcefield: str = "ff14SB",
    water_model: str = "tip3p",
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Parameterise all (crystal) water molecules in a given file using the specified
    forcefield and water model.

    Parameters
    ----------
    filename : str
        Path to the water coordinate file (e.g. water.pdb) containing water molecules.
    forcefield : str
        Force field name to apply (e.g. "ff14SB").
    water_model : str
        Identifier for the water model (e.g. "tip3p", "spce").
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised water molecules.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Water file not found: {file_path}")

    # empty or malformatted pdb file results in a BSS error
    waters_sys = BSS.IO.readMolecules(str(file_path))
    logger.info(
        f"Parameterising {len(waters_sys)} water molecule(s) with forcefield={forcefield}, \
            water_model={water_model}",
    )

    # Parameterise each water and extract the molecule
    parameterised = [
        BSS.Parameters.parameterise(
            molecule=wat,
            forcefield=forcefield,
            water_model=water_model,
        ).getMolecule()
        for wat in waters_sys
    ]

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    # Save the system to local files if output_dir and filename_stem are provided
    save_system_to_local(
        system=system,
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    return system


def _parameterise_ligand(
    file_path: str,
    forcefield: str = "openff_unconstrained-2.0.0",
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Parameterise the ligand. In a classic FEP calculation, we should only have one ligand

    Parameters
    ----------
    file_path : str
        Path to the ligand coordinate file (e.g. ligand.sdf) containing the ligand molecule.
    forcefield : str
        Force field name to apply (e.g. "openff_unconstrained-2.0.0").
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised ligand molecule.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ligand file not found: {file_path}")

    lig_sys = BSS.IO.readMolecules(file_path)
    logger.info(
        f"Parameterising {len(lig_sys)} ligand molecule(s) with forcefield={forcefield}",
    )

    # Parameterise each water and extract the molecule
    rename_lig(lig_sys, "LIG")  # Ensure that the ligand is named "LIG"
    parameterised = []
    for lig in lig_sys:
        lig_charge = round(lig.charge().value())
        if lig_charge != 0:
            logger.warning(
                f"Ligand {lig.__str__} has a charge of {lig_charge}. Co-alchemical ion"
                " approach will be used. Ensure that your box is large enough to avoid artefacts."
            )
        param_args = {"molecule": lig, "forcefield": forcefield}
        # Only include ligand charge if we're using gaff (OpenFF doesn't need it)
        if "gaff" in forcefield:
            param_args["net_charge"] = lig_charge

        param_lig = BSS.Parameters.parameterise(**param_args).getMolecule()
        parameterised.append(param_lig)

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    # Save the system to local files if output_dir and filename_stem are provided
    save_system_to_local(
        system=system,
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    return system


def _parameterise_protein(
    file_path: str,
    forcefield: str = "ff14SB",
    filename_stem: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> BSS._SireWrappers._system.System:
    """
    Parameterise the protein. In a classic FEP calculation, we should only have one protein

    Parameters
    ----------
    file_path : str
        Path to the protein coordinate file (e.g. "protein.pdb") containing the protein molecule.
    forcefield : str
        Force field name to apply (e.g. "ff14SB").
    filename_stem : Optional[str]
        The stem of the filename to use for the output files. Must be set if output_dir is set.
    output_dir : Optional[str]
        Directory to write the output files to. If None, not written to disk.

    Returns
    -------
    System
        A BioSimSpace System containing the assembled, parameterised protein molecule.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Protein file not found: {file_path}")

    protein_sys = BSS.IO.readMolecules(file_path)
    logger.info(
        f"Parameterising {len(protein_sys)} protein molecule(s) with forcefield={forcefield}",
    )

    # Parameterise each water and extract the molecule
    parameterised = [
        BSS.Parameters.parameterise(
            molecule=protein,
            forcefield=forcefield,
        ).getMolecule()
        for protein in protein_sys
    ]

    # Assemble into a single System
    system = parameterised[0].toSystem()
    for mol in parameterised[1:]:
        system += mol  # only works for different molecules

    # Save the system to local files if output_dir and filename_stem are provided
    save_system_to_local(
        system=system,
        filename_stem=filename_stem,
        output_dir=output_dir,
    )

    return system
