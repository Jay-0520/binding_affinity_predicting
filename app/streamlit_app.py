import json
import tempfile
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from binding_affinity_predicting.components.analysis.free_energy_estimators import (
    FreeEnergyEstimator,
)
from binding_affinity_predicting.components.analysis.uncorrelate_subsampler import (
    perform_uncorrelating_subsampling,
)

# Import your analysis modules
from binding_affinity_predicting.components.analysis.xvg_data_loader import (
    load_alchemical_data,
)


def main():
    st.set_page_config(page_title="Free Energy Analysis", page_icon="⚛️", layout="wide")

    st.title("⚛️ Free Energy Analysis Tool")
    st.markdown(
        "Upload GROMACS XVG files for correlation analysis and free energy estimation"
    )

    # Sidebar for global parameters
    st.sidebar.header("Analysis Parameters")
    temperature = st.sidebar.number_input(
        "Temperature (K)", value=298.15, min_value=0.0, step=0.1
    )
    units = st.sidebar.selectbox("Output Units", ["kcal", "kJ", "kBT"])
    skip_time = st.sidebar.number_input(
        "Skip Time (ps)", value=0.0, min_value=0.0, step=1.0
    )

    # Main content area with tabs
    tab1, tab2 = st.tabs(
        ["📁 Data Upload & Correlation Analysis", "🧮 Free Energy Estimation"]
    )

    with tab1:
        correlation_analysis_tab(temperature, skip_time)

    with tab2:
        free_energy_estimation_tab(temperature, units)


def correlation_analysis_tab(temperature: float, skip_time: float):
    """Tab for data upload and correlation analysis."""
    st.header("Correlation Analysis")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload XVG files (one per lambda state)",
        type=['xvg'],
        accept_multiple_files=True,
        help="Upload GROMACS XVG files containing dH/dλ and cross-evaluation data",
    )

    if not uploaded_files:
        st.info("Please upload XVG files to begin analysis")
        return

    # Save uploaded files temporarily
    temp_files = []
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xvg') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_files.append(Path(tmp_file.name))

    try:
        # Load alchemical data
        with st.spinner("Loading alchemical data..."):
            alchemical_data = load_alchemical_data(
                xvg_files=temp_files,
                skip_time=skip_time,
                temperature=temperature,
                reduce_to_dimensionless=True,
            )

        # Display data summary
        st.success(f"✅ Successfully loaded data from {len(uploaded_files)} files")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lambda States", alchemical_data['lambda_vectors'].shape[0])
        with col2:
            st.metric("Components", alchemical_data['lambda_vectors'].shape[1])
        with col3:
            st.metric("Max Snapshots", alchemical_data['dhdl_timeseries'].shape[2])

        # Lambda vectors display
        st.subheader("Lambda Vectors")
        lambda_df = pd.DataFrame(
            alchemical_data['lambda_vectors'], columns=['Coulomb', 'VdW', 'Bonded']
        )
        lambda_df.index.name = 'State'
        st.dataframe(lambda_df, use_container_width=True)

        # Correlation analysis parameters
        st.subheader("Correlation Analysis Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            observable = st.selectbox(
                "Observable for correlation",
                ["dhdl", "dhdl_all", "de"],
                help="dhdl: changing components only, dhdl_all: all components, "
                "de: energy differences",
            )

        with col2:
            min_uncorr_samples = st.number_input(
                "Min uncorrelated samples", value=50, min_value=1, step=1
            )

        with col3:
            fast_analysis = st.checkbox("Fast analysis", value=False)

        # Perform correlation analysis
        if st.button("🔍 Perform Correlation Analysis", type="primary"):
            perform_correlation_analysis(
                alchemical_data, observable, min_uncorr_samples, fast_analysis
            )

    except Exception as e:
        st.error(f"Error processing files: {str(e)}")

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass


def perform_correlation_analysis(
    alchemical_data: dict, observable: str, min_uncorr_samples: int, fast_analysis: bool
):
    """Perform and display correlation analysis results."""

    with st.spinner("Performing correlation analysis..."):
        # Calculate start and end indices (skip equilibration)
        start_indices = np.zeros(len(alchemical_data['lambda_vectors']), dtype=int)
        end_indices = alchemical_data['nsnapshots']

        # Perform uncorrelated subsampling
        dhdl_uncorr, potential_uncorr, num_uncorr_samples = (
            perform_uncorrelating_subsampling(
                dhdl_timeseries=alchemical_data['dhdl_timeseries'],
                lambda_vectors=alchemical_data['lambda_vectors'],
                start_indices=start_indices,
                end_indices=end_indices,
                potential_energies=alchemical_data['potential_energies'],
                observable=observable,
                min_uncorr_samples=min_uncorr_samples,
                fast_analysis=fast_analysis,
            )
        )

    # Store results in session state for use in free energy estimation
    st.session_state['alchemical_data'] = alchemical_data
    st.session_state['dhdl_uncorr'] = dhdl_uncorr
    st.session_state['potential_uncorr'] = potential_uncorr
    st.session_state['num_uncorr_samples'] = num_uncorr_samples

    # Display results
    st.success("✅ Correlation analysis completed!")

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Uncorr. Samples", int(num_uncorr_samples.sum()))
    with col2:
        efficiency = (
            num_uncorr_samples.sum() / alchemical_data['nsnapshots'].sum()
        ) * 100
        st.metric("Sampling Efficiency", f"{efficiency:.1f}%")
    with col3:
        avg_corr_time = alchemical_data['nsnapshots'].sum() / num_uncorr_samples.sum()
        st.metric("Avg. Correlation Time", f"{avg_corr_time:.1f}")

    # Uncorrelated samples per state
    st.subheader("Uncorrelated Samples per State")
    samples_df = pd.DataFrame(
        {
            'State': range(len(num_uncorr_samples)),
            'Total Samples': alchemical_data['nsnapshots'],
            'Uncorr. Samples': num_uncorr_samples,
            'Efficiency (%)': (num_uncorr_samples / alchemical_data['nsnapshots'])
            * 100,
        }
    )
    st.dataframe(samples_df, use_container_width=True)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Samples comparison
    x = np.arange(len(num_uncorr_samples))
    ax1.bar(x - 0.2, alchemical_data['nsnapshots'], 0.4, label='Total', alpha=0.7)
    ax1.bar(x + 0.2, num_uncorr_samples, 0.4, label='Uncorrelated', alpha=0.7)
    ax1.set_xlabel('Lambda State')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Sample Count Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Efficiency
    efficiency_per_state = (num_uncorr_samples / alchemical_data['nsnapshots']) * 100
    ax2.plot(x, efficiency_per_state, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Lambda State')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Sampling Efficiency per State')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def free_energy_estimation_tab(temperature: float, units: str):
    """Tab for free energy estimation using various methods."""
    st.header("Free Energy Estimation")

    # Check if correlation analysis has been performed
    if 'alchemical_data' not in st.session_state:
        st.warning("⚠️ Please perform correlation analysis first in the previous tab")
        return

    # Get data from session state
    alchemical_data = st.session_state['alchemical_data']
    dhdl_uncorr = st.session_state.get('dhdl_uncorr')
    potential_uncorr = st.session_state.get('potential_uncorr')
    num_uncorr_samples = st.session_state['num_uncorr_samples']

    # Method selection
    st.subheader("Select Free Energy Methods")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Thermodynamic Integration**")
        use_ti_trap = st.checkbox("TI Trapezoidal", value=True)
        use_ti_cubic = st.checkbox("TI Cubic Spline", value=True)

    with col2:
        st.write("**Bennett Acceptance Ratio**")
        use_bar = st.checkbox("BAR", value=True)
        use_mbar = st.checkbox("MBAR", value=True)
        use_ubar = st.checkbox("UBAR", value=False)
        use_rbar = st.checkbox("RBAR", value=False)

    with col3:
        st.write("**Exponential Averaging**")
        use_dexp = st.checkbox("DEXP", value=True)
        use_iexp = st.checkbox("IEXP", value=True)
        use_gdel = st.checkbox("GDEL", value=False)
        use_gins = st.checkbox("GINS", value=False)

    # Build methods list
    methods = []
    if use_ti_trap:
        methods.append('TI_trapezoidal')
    if use_ti_cubic:
        methods.append('TI_cubic')
    if use_bar:
        methods.append('BAR')
    if use_mbar:
        methods.append('MBAR')
    if use_ubar:
        methods.append('UBAR')
    if use_rbar:
        methods.append('RBAR')
    if use_dexp:
        methods.append('DEXP')
    if use_iexp:
        methods.append('IEXP')
    if use_gdel:
        methods.append('GDEL')
    if use_gins:
        methods.append('GINS')

    if not methods:
        st.warning("Please select at least one method")
        return

    # Additional parameters
    with st.expander("Advanced Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            relative_tolerance = st.number_input(
                "Relative Tolerance (BAR/MBAR)",
                value=1e-10,
                format="%.2e",
                min_value=1e-15,
                max_value=1e-3,
            )
            verbose = st.checkbox("Verbose Output", value=False)
        with col2:
            trial_range_min = st.number_input("RBAR Trial Range Min", value=-10, step=1)
            trial_range_max = st.number_input("RBAR Trial Range Max", value=10, step=1)

    # Perform free energy estimation
    if st.button("🧮 Calculate Free Energies", type="primary"):
        perform_free_energy_estimation(
            alchemical_data=alchemical_data,
            dhdl_uncorr=dhdl_uncorr,
            potential_uncorr=potential_uncorr,
            num_uncorr_samples=num_uncorr_samples,
            methods=methods,
            temperature=temperature,
            units=units,
            relative_tolerance=relative_tolerance,
            verbose=verbose,
            trial_range=(trial_range_min, trial_range_max),
        )


def perform_free_energy_estimation(
    alchemical_data: dict,
    dhdl_uncorr: Optional[np.ndarray],
    potential_uncorr: Optional[np.ndarray],
    num_uncorr_samples: np.ndarray,
    methods: List[str],
    temperature: float,
    units: str,
    relative_tolerance: float,
    verbose: bool,
    trial_range: tuple,
):
    """Perform free energy estimation and display results."""

    with st.spinner("Calculating free energies..."):
        # Initialize estimator
        estimator = FreeEnergyEstimator(temperature=temperature, units=units)

        # Calculate dH/dλ statistics for TI methods
        ave_dhdl = None
        std_dhdl = None
        if dhdl_uncorr is not None:
            ave_dhdl = np.mean(dhdl_uncorr, axis=2)
            std_dhdl = np.std(dhdl_uncorr, axis=2, ddof=1) / np.sqrt(
                dhdl_uncorr.shape[2]
            )

        # Estimate free energies
        results = estimator.estimate_all_methods(
            potential_energies=potential_uncorr,
            sample_counts=num_uncorr_samples,
            lambda_vectors=alchemical_data['lambda_vectors'],
            ave_dhdl=ave_dhdl,
            std_dhdl=std_dhdl,
            methods=methods,
            relative_tolerance=relative_tolerance,
            verbose=verbose,
            trial_range=trial_range,
        )

    # Display results
    st.success("✅ Free energy calculations completed!")

    # Create results summary
    results_data = []
    for method, result in results.items():
        if result.get('success', False):
            results_data.append(
                {
                    'Method': method,
                    'Free Energy': f"{result.get('free_energy', 0.0):.3f}",
                    'Error': f"{result.get('error', 0.0):.3f}",
                    'Units': units,
                    'Success': '✅',
                }
            )
        else:
            results_data.append(
                {
                    'Method': method,
                    'Free Energy': 'Failed',
                    'Error': 'Failed',
                    'Units': units,
                    'Success': '❌',
                }
            )

    # Display results table
    st.subheader("Free Energy Results")
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

    # Statistics and visualization
    successful_results = [
        (method, result)
        for method, result in results.items()
        if result.get('success', False)
    ]

    if successful_results:
        # Calculate statistics
        free_energies = [result['free_energy'] for _, result in successful_results]
        errors = [result['error'] for _, result in successful_results]
        method_names = [method for method, _ in successful_results]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean ΔG", f"{np.mean(free_energies):.3f} {units}")
        with col2:
            st.metric("Std Dev", f"{np.std(free_energies):.3f} {units}")
        with col3:
            st.metric(
                "Range", f"{np.max(free_energies) - np.min(free_energies):.3f} {units}"
            )

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Free energies with error bars
        y_pos = np.arange(len(method_names))
        ax1.barh(y_pos, free_energies, xerr=errors, capsize=5, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(method_names)
        ax1.set_xlabel(f'Free Energy ({units})')
        ax1.set_title('Free Energy Estimates by Method')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error comparison
        ax2.bar(method_names, errors, alpha=0.7, color='orange')
        ax2.set_ylabel(f'Error ({units})')
        ax2.set_title('Error Estimates by Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Download results
        st.subheader("Export Results")
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📥 Download Results (JSON)",
            data=results_json,
            file_name=f"free_energy_results_{temperature}K.json",
            mime="application/json",
        )

    else:
        st.error("❌ All methods failed. Please check your data and parameters.")


if __name__ == "__main__":
    main()
