"""
GROMACS XVG File Parser for Alchemical Free Energy Calculations

This module provides functions to parse GROMACS .xvg files containing:
- dH/dλ time series data
- Cross-evaluation energy differences (ΔH)
- Lambda state information

Supports the data format needed for alchemical free energy analysis with
methods like MBAR, BAR, and thermodynamic integration.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from binding_affinity_predicting.components.analysis.utils import (
    calculate_beta_parameter,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to avoid debug logs by default


# TODO: maybe use this in gromacs_orchestration.py as well?
class LambdaState:
    """Represents a lambda state with its parameters."""

    def __init__(self, coul: float, vdw: float, bonded: float, state_id: int):
        self.coul = coul
        self.vdw = vdw
        self.bonded = bonded
        self.state_id = state_id

    def __repr__(self):
        return f"LambdaState({self.state_id}: coul={self.coul}, vdw={self.vdw}, bonded={self.bonded})"  # noqa: E501

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for use in algorithms."""
        return np.array([self.coul, self.vdw, self.bonded])


class GromacsXVGParser:
    """
    Parser for GROMACS XVG files with alchemical free energy data.

    An example XVG file contains:

        # This file was created Sun Jun  8 00:35:29 2025
        # Created by:
        #                      :-) GROMACS - gmx mdrun, 2024.4 (-:
        #
        # Executable:   /usr/local/gromacs/bin/gmx
        # Data prefix:  /usr/local/gromacs
        # Working dir:  ~/output/bound/lambda_1/run_1
        # Command line:
        #   gmx mdrun -deffnm lambda_1_run_1 -cpi lambda_1_run_1.cpt
        # gmx mdrun is part of G R O M A C S:
        #
        # Gromacs Runs On Most of All Computer Systems
        #
        @    title "dH/dλ and λ"
        @    xaxis  label "Time (ps)"
        @    yaxis  label "dH/dλ and λ (kJ mol⁻¹ [λ]⁻¹)"
        @TYPE xy
        @ subtitle "T = 300 (K) λ state 1: (coul-lambda, vdw-lambda, bonded-lambda) = (0.0000, 0.1000, 0.2500)"  # noqa: E501
        @ view 0.15, 0.15, 0.75, 0.85
        @ legend on
        @ legend box on
        @ legend loctype view
        @ legend 0.78, 0.8
        @ legend length 2
        @ s0 legend "dH/dλ coul-lambda = 0.0000"
        @ s1 legend "dH/dλ vdw-lambda = 0.1000"
        @ s2 legend "dH/dλ bonded-lambda = 0.2500"
        @ s3 legend "ΔH*λ to (0.0000, 0.0000, 0.0000)"  # NOTE: we must set calc-lambda-neighbors = -1 in mdp to get this  # noqa: E501
        @ s4 legend "ΔH*λ to (0.0000, 0.1000, 0.2500)"
        @ s5 legend "pV (kJ/mol)"
        0.0000 10.016686 24.389038 0.0000000 -0.98744838 0.0000000 21.151768
        0.2000 5.0095649 -82.750069 0.0000000 13.366991 0.0000000 19.273001
        0.4000 5.3747883 -181.20355 0.0000000 31.995344 0.0000000 18.717262
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset parser state."""
        self.lambda_states: list[LambdaState] = []
        self.current_state: Optional[LambdaState] = None
        self.dhdl_components: list[str] = []
        self.cross_eval_states: list[LambdaState] = []

    def parse_xvg_file(self, file_path: Path, skip_time: float = 0.0) -> dict:
        """
        Parse complete GROMACS XVG file with dH/dλ and cross-evaluation data.

        Parameters:
        -----------
        file_path : Path
            Path to GROMACS .xvg file
        skip_time : float
            Time to skip for equilibration (in ps)

        Returns:
        --------
        Dict containing:
            - 'times': np.ndarray - Time points (ps)
            - 'current_state': LambdaState - Lambda state of this simulation
            - 'dhdl_components': Dict[str, np.ndarray] - dH/dλ for each component
            - 'cross_evaluations': Dict[int, np.ndarray] - ΔH to each target state
            - 'total_energy': np.ndarray - Total energy time series (if available)
            - 'pV': np.ndarray - pV term (if available)
        """
        self.reset()

        # Parse header to extract lambda state information
        self._parse_header(file_path)

        # Parse data section
        data = self._parse_data_section(file_path, skip_time)

        logger.debug(f"Parsed XVG file {file_path.name}:")
        logger.debug(f"  Current state: {self.current_state}")
        logger.debug(f"  Data points: {len(data['times'])}")
        logger.debug(f"  dH/dλ components: {len(data['dhdl_components'])}")
        logger.debug(f"  Cross-evaluations: {len(data['cross_evaluations'])}")

        return data

    def _parse_header(self, file_path: Path) -> None:
        """Parse XVG header to extract lambda state and column information."""
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('@ subtitle'):
                    # Extract current lambda state from subtitle
                    self._parse_current_state(line)
                elif line.startswith('@ s') and 'legend' in line:
                    # Parse column legends
                    self._parse_legend(line)

    def _parse_current_state(self, subtitle_line: str) -> None:
        """Extract current lambda state from subtitle line."""
        # Example: @ subtitle "T = 300 (K) \\xl\\f{} state 13: (coul-lambda,
        # vdw-lambda, bonded-lambda) = (0.7500, 0.0000, 1.0000)"
        pattern = r'state (\d+):.*?=\s*\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)'
        match = re.search(pattern, subtitle_line)

        if match:
            state_id = int(match.group(1))
            coul = float(match.group(2))
            vdw = float(match.group(3))
            bonded = float(match.group(4))

            self.current_state = LambdaState(coul, vdw, bonded, state_id)
            logger.debug(f"Found current state: {self.current_state}")
        else:
            raise ValueError(
                f"Could not derive lambda state from subtitle: {subtitle_line} "
                f"check XVG file format"
            )

    def _parse_legend(self, legend_line: str) -> None:
        """Parse legend lines to identify column types."""
        if ('dH/d\\xl\\f{}' in legend_line) or ('dH/dλ' in legend_line):
            # dH/dλ component
            if 'coul-lambda' in legend_line:
                self.dhdl_components.append('coulomb')
            elif 'vdw-lambda' in legend_line:
                self.dhdl_components.append('vdw')
            elif 'bonded-lambda' in legend_line:
                self.dhdl_components.append('bonded')

        elif ('\\xD\\f{}H \\xl\\f{} to' in legend_line) or ('ΔH λ to' in legend_line):
            # Cross-evaluation to target state
            # Example: @ s4 legend "\\xD\\f{}H \\xl\\f{} to (0.0000, 0.0000, 0.0000)"
            pattern = r'to \(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)'
            match = re.search(pattern, legend_line)

            if match:
                coul = float(match.group(1))
                vdw = float(match.group(2))
                bonded = float(match.group(3))
                state_id = len(self.cross_eval_states)  # Assign sequential ID

                target_state = LambdaState(coul, vdw, bonded, state_id)
                self.cross_eval_states.append(target_state)
                logger.debug(f"Found cross-evaluation target: {target_state}")
            else:
                raise ValueError(
                    f"Could not parse cross-evaluation energy from legend: {legend_line} "
                    "check XVG file format or set calc-lambda-neighbors = -1 in mdp file"
                )

    def _parse_data_section(self, file_path: Path, skip_time: float) -> dict:
        """Parse the data section of the XVG file."""
        times = []
        total_energies = []
        dhdl_data: dict[str, list[float]] = {comp: [] for comp in self.dhdl_components}
        cross_eval_data: dict[int, list[float]] = {
            i: [] for i in range(len(self.cross_eval_states))
        }
        pv_data = []

        with open(file_path, 'r') as f:
            # Parse data lines
            for line in f:
                line = line.strip()

                # Skip comments
                if line.startswith('#') or line.startswith('@'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    time = float(parts[0])
                    if time < skip_time:
                        continue

                    times.append(time)
                    # Parse columns based on expected structure
                    col_idx = 1
                    # Total energy (column 1)
                    if col_idx < len(parts):
                        total_energies.append(float(parts[col_idx]))
                        col_idx += 1

                    # dH/dλ components (columns 2, 3, 4)
                    for comp in self.dhdl_components:
                        if col_idx < len(parts):
                            dhdl_data[comp].append(float(parts[col_idx]))
                            col_idx += 1

                    # Cross-evaluations (remaining columns except last)
                    cross_eval_energies = []
                    for i in range(len(self.cross_eval_states)):
                        if col_idx < len(parts):
                            cross_eval_energies.append(float(parts[col_idx]))
                            col_idx += 1

                    # pV term (last column) and it must be present
                    if col_idx < len(parts):
                        pv_term = float(parts[col_idx])
                        pv_data.append(pv_term)

                    # Add pV to cross-evaluations like the original parser
                    # This matches: u_klt = P.beta * ( data[r1:r2, :] + data[-1,:] )
                    for i, cross_eval_energy in enumerate(cross_eval_energies):
                        cross_eval_data[i].append(cross_eval_energy + pv_term)

                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Skipping malformed line: {line[:50]}... Error: {e}"
                    )
                    continue

        # Convert to numpy arrays
        result = {
            'times': np.array(times),
            'current_state': self.current_state,
            'dhdl_components': {
                comp: np.array(dhdl_data[comp]) for comp in self.dhdl_components
            },
            'cross_evaluations': {
                i: np.array(cross_eval_data[i]) for i in cross_eval_data
            },
            'total_energy': np.array(total_energies) if total_energies else None,
            'pV': np.array(pv_data) if pv_data else None,
        }

        return result


def load_alchemical_data(
    xvg_files: Sequence[Union[Path, str]],
    skip_time: float = 0.0,
    temperature: float = 298.15,
    reduce_to_dimensionless: bool = True,
    save_to_path: Optional[Path] = None,
) -> dict[str, np.ndarray]:
    """
    Load data (required for free energy estimators) from multiple GROMACS XVG files.

    Parameters:
    -----------
    xvg_files : List[Path]
        List of paths to GROMACS .xvg files (one per lambda state)
    skip_time : float
        Time to skip for equilibration (in ps)
    temperature : float, default 300.0
        Temperature in Kelvin for beta calculation
    reduce_to_dimensionless: bool, default True
        If True, reduce potential energies to dimensionless values using beta
    save_to_path: Optional[Path]
        If provided, save the parsed data to this path as a .pickle file

    Returns:
    --------
    dict[str, np.ndarray]
        Dictionary containing:
        dhdl_timeseries : np.ndarray, shape (num_states, num_components, max_snapshots)
            Time series of dH/dλ values
        potential_energies : np.ndarray, shape (num_states, num_states, max_snapshots)
            Cross-evaluation energy differences (must be reduced and dimensionless):
             - the total potential energy of the current configuration at every other
             λ‐combination in the calculation.
        lambda_vectors : np.ndarray, shape (num_states, num_components)
            Lambda parameter values for each state
        nsnapshots : np.ndarray, shape (num_states,)
            Number of equilibrated snapshots per lambda state
    """
    parser = GromacsXVGParser()
    parsed_files = []

    # Parse all XVG files
    for file_path in xvg_files:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            raise FileNotFoundError(f"File {file_path} does not exist")

        try:
            data = parser.parse_xvg_file(file_path, skip_time)
            parsed_files.append(data)
            logger.debug(f"Successfully parsed {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            continue

    if not parsed_files:
        raise ValueError("No valid XVG files could be parsed")

    # Determine dimensions
    num_states = len(parsed_files)

    # Get component names from first file
    first_file = parsed_files[0]
    component_names = list(first_file['dhdl_components'].keys())
    num_components = len(component_names)

    # Find maximum number of snapshots
    max_snapshots = max(len(data['times']) for data in parsed_files)

    logger.debug(
        f"Dataset dimensions: {num_states} states, {num_components} "
        f"components, {max_snapshots} max snapshots"
    )

    # Initialize output arrays
    dhdl_timeseries = np.zeros((num_states, num_components, max_snapshots))
    potential_energies = np.zeros((num_states, num_states, max_snapshots))
    lambda_vectors = np.zeros((num_states, num_components))
    nsnapshots = np.zeros(num_states, dtype=int)

    # Fill arrays with data
    for state_idx, data in enumerate(parsed_files):
        current_state = data['current_state']
        num_snapshots = len(data['times'])

        # Store actual number of snapshots for this state
        nsnapshots[state_idx] = num_snapshots

        # Store lambda vector
        lambda_vectors[state_idx] = current_state.to_vector()

        # Store dH/dλ data
        for comp_idx, comp_name in enumerate(component_names):
            if comp_name in data['dhdl_components']:
                dhdl_timeseries[state_idx, comp_idx, :num_snapshots] = data[
                    'dhdl_components'
                ][comp_name]

        # Store cross-evaluation data
        cross_evals = data['cross_evaluations']
        for target_idx in cross_evals:
            if target_idx < num_states:  # Only store if target state exists
                potential_energies[state_idx, target_idx, :num_snapshots] = cross_evals[
                    target_idx
                ]

    # Reduce potential energies to dimensionless values
    if reduce_to_dimensionless:
        beta = calculate_beta_parameter(
            temperature=temperature, units='kJ', software='Gromacs'
        )
        potential_energies *= beta  # Convert to dimensionless
        dhdl_timeseries *= beta  # Convert dH/dλ to dimensionless
    else:
        logger.warning(
            "Potential energies will not be reduced to dimensionless values. "
            "This may affect free energy calculations."
        )

    if save_to_path:
        # Save parsed data to a pickle file
        import pickle

        with open(save_to_path, 'wb') as f:
            pickle.dump(
                {
                    'dhdl_timeseries': dhdl_timeseries,
                    'potential_energies': potential_energies,
                    'lambda_vectors': lambda_vectors,
                    'nsnapshots': nsnapshots,
                },
                f,
            )
        logger.info(f"Saved parsed data to {save_to_path}")

    logger.info("Successfully loaded alchemical data arrays")
    return {
        'dhdl_timeseries': dhdl_timeseries,
        'potential_energies': potential_energies,
        'lambda_vectors': lambda_vectors,
        'nsnapshots': nsnapshots,
    }
