#!/usr/bin/env python

######################################################################
# Alchemical Analysis: An open tool implementing some recommended practices for analyzing alchemical free energy calculations
# Copyright 2011-2015 UC Irvine and the Authors
#
# Authors: Pavel Klimovich, Michael Shirts and David Mobley
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, see <http://www.gnu.org/licenses/>.
######################################################################

# ===================================================================================================
# IMPORTS
# ===================================================================================================

## Not a built-in module. Will be called from main, whenever needed. ##
## import pymbar      Multistate Bennett Acceptance Ratio estimator. ##

import os  # for os interface
import pdb  # for debugging
import pickle  # for full-precision data storage
import sys
import time as ttt_time  # for timing
from optparse import OptionParser  # for parsing command-line options

import numpy

# ===================================================================================================
# INPUT OPTIONS
# ===================================================================================================

parser = OptionParser()
parser.add_option(
    '-a',
    '--software',
    dest='software',
    help='Package\'s name the data files come from: Gromacs, Sire, Desmond, or AMBER. Default: Gromacs.',
    default='Gromacs',
)
parser.add_option(
    '-b',
    '--fraction',
    dest='fraction',
    help='The fraction of the energy file will be used to calculate the statistics. Default: 1.0',
    default=1.0,
    type=float,
)
parser.add_option(
    '-c',
    '--cfm',
    dest='bCFM',
    help='The Curve-Fitting-Method-based consistency inspector. Default: False.',
    default=False,
    action='store_true',
)
parser.add_option(
    '-d',
    '--dir',
    dest='datafile_directory',
    help='Directory in which data files are stored. Default: Current directory.',
    default='.',
)
parser.add_option(
    '-e',
    '--backward',
    dest='backward',
    help='Extract the energy data from the backward direction. Default: False',
    default=False,
    action='store_true',
)
parser.add_option(
    '-f',
    '--forwrev',
    dest='bForwrev',
    help='Plot the free energy change as a function of time in both directions, with the specified number of points in the time plot. The number of time points (an integer) must be provided. Default: 0',
    default=0,
    type=int,
)
parser.add_option(
    '-g',
    '--breakdown',
    dest='breakdown',
    help='Plot the free energy differences evaluated for each pair of adjacent states for all methods, including the dH/dlambda curve for TI. Default: False.',
    default=False,
    action='store_true',
)
parser.add_option(
    '-i',
    '--threshold',
    dest='uncorr_threshold',
    help='Proceed with correlated samples if the number of uncorrelated samples is found to be less than this number. If 0 is given, the time series analysis will not be performed at all. Default: 50.',
    default=50,
    type=int,
)
parser.add_option(
    '-j',
    '--resultfilename',
    dest='resultfilename',
    help='custom defined result filename prefix. Default: results',
    default='results',
)
parser.add_option(
    '-k',
    '--koff',
    dest='bSkipLambdaIndex',
    help='Give a string of lambda indices separated by \'-\' and they will be removed from the analysis. (Another approach is to have only the files of interest present in the directory). Default: None.',
    default='',
)
parser.add_option(
    '-m',
    '--methods',
    dest='methods',
    help='A list of the methods to esitimate the free energy with. Default: [TI, TI-CUBIC, DEXP, IEXP, BAR, MBAR]. To add/remove methods to the above list provide a string formed of the method strings preceded with +/-. For example, \'-ti_cubic+gdel\' will turn methods into [TI, DEXP, IEXP, BAR, MBAR, GDEL]. \'ti_cubic+gdel\', on the other hand, will call [TI-CUBIC, GDEL]. \'all\' calls the full list of supported methods [TI, TI-CUBIC, DEXP, IEXP, GINS, GDEL, BAR, UBAR, RBAR, MBAR].',
    default='',
)
parser.add_option(
    '-n',
    '--uncorr',
    dest='uncorr',
    help='The observable to be used for the autocorrelation analysis; either \'dhdl_all\' (obtained as a sum over all energy components) or \'dhdl\' (obtained as a sum over those energy components that are changing; default) or \'dE\'. In the latter case the energy differences dE_{i,i+1} (dE_{i,i-1} for the last lambda) are used.',
    default='dhdl',
)
parser.add_option(
    '-o',
    '--out',
    dest='output_directory',
    help='Directory in which the output files produced by this script will be stored. Default: Same as datafile_directory.',
    default='',
)
parser.add_option(
    '-p',
    '--prefix',
    dest='prefix',
    help='Prefix for datafile sets, i.e.\'dhdl\' (default).',
    default='dhdl',
)
parser.add_option(
    '-q',
    '--suffix',
    dest='suffix',
    help='Suffix for datafile sets, i.e. \'xvg\' (default).',
    default='xvg',
)
parser.add_option(
    '-r',
    '--decimal',
    dest='decimal',
    help='The number of decimal places the free energies are to be reported with. No worries, this is for the text output only; the full-precision data will be stored in \'results.pickle\'. Default: 3.',
    default=3,
    type=int,
)
parser.add_option(
    '-s',
    '--skiptime',
    dest='equiltime',
    help='Discard data prior to this specified time as \'equilibration\' data. Units picoseconds. Default: 0 ps.',
    default=0,
    type=float,
)
parser.add_option(
    '-t',
    '--temperature',
    dest='temperature',
    help="Temperature in K. Default: 298 K.",
    default=298,
    type=float,
)
parser.add_option(
    '-u',
    '--units',
    dest='units',
    help='Units to report energies: \'kJ\', \'kcal\', and \'kBT\'. Default: \'kJ\'',
    default='kJ',
)
parser.add_option(
    '-v',
    '--verbose',
    dest='verbose',
    help='Verbose option. Default: False.',
    default=False,
    action='store_true',
)
parser.add_option(
    '-w',
    '--overlap',
    dest='overlap',
    help='Print out and plot the overlap matrix. Default: False.',
    default=False,
    action='store_true',
)
parser.add_option(
    '-x',
    '--ignoreWL',
    dest='bIgnoreWL',
    help='Do not check whether the WL weights are equilibrated. No log file needed as an accompanying input.',
    default=False,
    action='store_true',
)
parser.add_option(
    '-y',
    '--tolerance',
    dest='relative_tolerance',
    help="Convergence criterion for the energy estimates with BAR and MBAR. Default: 1e-10.",
    default=1e-10,
    type=float,
)
parser.add_option(
    '-z',
    '--initialize',
    dest='init_with',
    help='The initial MBAR free energy guess; either \'BAR\' or \'zeros\'. Default: \'BAR\'.',
    default='BAR',
)

import os
import sys

import numpy
import pymbar


def zero_output(K, P):
    """Adapted from totalEnergies() function to write out all zeros in results.txt file"""
    print("Generating results.txt with all 0")
    df_allk = list()
    ddf_allk = list()
    for k in range(K - 1):
        df = dict()
        ddf = dict()
        for name in P.methods:
            df[name] = 0.0
            ddf[name] = 0.0
        df_allk = numpy.append(df_allk, df)
        ddf_allk = numpy.append(ddf_allk, ddf)
    segments = ['Coulomb', 'vdWaals', 'TOTAL']

    def printLine(str1, str2, d1=None, d2=None):
        text = str1
        for name in P.methods:
            if d1 == 'plain':
                text += ' ' + str2
            if d1 == 'name':
                text += ' ' + str2 % (name, P.units)
            if d1 and d2:
                text += ' ' + str2 % (
                    d1[name] / P.beta_report,
                    d2[name] / P.beta_report,
                )
        outtext.append(text + '\n')
        return

    d = P.decimal
    str_dash = (d + 7 + 6 + d + 2) * '-'
    str_dat = ('X%d.%df  +-  X%d.%df' % (d + 7, d, d + 2, d)).replace('X', '%')
    str_names = ('X%ds X-%ds' % (d + 6, d + 8)).replace('X', '%')
    outtext = []
    printLine(12 * '-', str_dash, 'plain')
    printLine('%-12s' % '   States', str_names, 'name')
    printLine(12 * '-', str_dash, 'plain')
    for k in range(K - 1):
        printLine('%4d -- %-4d' % (k, k + 1), str_dat, df_allk[k], ddf_allk[k])
    printLine(12 * '-', str_dash, 'plain')
    w = 12 + (1 + len(str_dash)) * len(P.methods)
    str_align = '{:I^%d}' % w
    if len(P.lv_names) > 1:
        for i in range(len(segments)):
            printLine('%9s:  ' % segments[i], str_dat, df_allk[i], ddf_allk[i])
    else:
        printLine('%9s:  ' % segments[-1], str_dat, 0.000, 0.000)
    # Store results.
    outfile = open(os.path.join(P.output_directory, 'results.txt'), 'w')
    outfile.write('# Command line was: %s\n\n' % ' '.join(sys.argv))
    outfile.writelines(outtext)
    outfile.close()


def zero_dFt(K, P, nsnapshots):

    print("Generating dF_t.txt with all 0")

    # Define a list of bForwrev equidistant time frames at which the free energy is to be estimated; count up the snapshots embounded between the time frames.
    n_tf = P.bForwrev + 1
    nss_tf = numpy.zeros([n_tf, K], int)
    increment = 1.0 / (n_tf - 1)
    if P.bExpanded:
        from collections import Counter  # for counting elements in an array

        tf = numpy.linspace(0.0, 1.0, n_tf) * (numpy.sum(nsnapshots) - 1) + 1
        tf[0] = 0
        for i in range(n_tf - 1):
            nss = Counter(extract_states[tf[i] : tf[i + 1]])
            nss_tf[i + 1] = numpy.array([nss[j] for j in range(K)])
    else:
        tf = numpy.linspace(0.0, 1.0, n_tf) * (max(nsnapshots) - 1) + 1
        tf[0] = 0
        for i in range(n_tf - 1):
            nss_tf[i + 1] = numpy.array(
                [min(j, tf[i + 1]) for j in nsnapshots]
            ) - numpy.sum(nss_tf[: i + 1], axis=0)
    # Define the real time scale (in ps) rather than a snapshot sequence.
    ts = ["%.1f" % ((i - (i != tf[0])) * P.snap_size[0] + P.equiltime) for i in tf]
    # Initialize arrays to store data points to be plotted.
    F_df = numpy.zeros(n_tf - 1, float)
    F_ddf = numpy.zeros(n_tf - 1, float)
    R_df = numpy.zeros(n_tf - 1, float)
    R_ddf = numpy.zeros(n_tf - 1, float)

    outtext = [
        "%12s %10s %-10s %17s %10s %s\n"
        % ('Time (ps)', 'Forward', P.units, 'Time (ps)', 'Reverse', P.units)
    ]
    outtext += [
        "%10s %11.3f +- %5.3f %18s %11.3f +- %5.3f\n"
        % (ts[1:][i], F_df[i], F_ddf[i], ts[:-1][i], R_df[i], R_ddf[i])
        for i in range(len(F_df))
    ]
    outfile = open(os.path.join(P.output_directory, 'dF_t.txt'), 'w')
    outfile.writelines(outtext)
    outfile.close()


# ===================================================================================================
# FUNCTIONS: Miscellanea.
# ===================================================================================================


def getMethods(string):
    """Returns a list of the methods the free energy is to be estimated with."""

    all_methods = [
        'TI',
        'TI-CUBIC',
        'DEXP',
        'IEXP',
        'GINS',
        'GDEL',
        'BAR',
        'UBAR',
        'RBAR',
        'MBAR',
    ]
    methods = ['TI', 'TI-CUBIC', 'DEXP', 'IEXP', 'BAR', 'MBAR']
    if (numpy.array(['Sire']) == P.software.title()).any():
        methods = ['TI', 'TI-CUBIC']
    if not string:
        return methods

    def addRemove(string):
        operation = string[0]
        string = string[1:] + '+'
        method = ''
        for c in string:
            if c.isalnum():
                method += c
            elif c == '_':
                method += '-'
            elif c == '-' or c == '+':
                if method in all_methods:
                    if operation == '-':
                        if method in methods:
                            methods.remove(method)
                    else:
                        if not method in methods:
                            methods.append(method)
                    method = ''
                    operation = c
                else:
                    parser.error(
                        "\nThere is no '%s' in the list of supported methods." % method
                    )
            else:
                parser.error(
                    "\nUnknown character '%s' in the method string is found." % c
                )
        return

    if string == 'ALL':
        methods = all_methods
    else:
        primo = string[0]
        if primo.isalpha():
            methods = string.replace('+', ' ').replace('_', '-').split()
            methods = [m for m in methods if m in all_methods]
        elif primo == '+' or primo == '-':
            addRemove(string)
        else:
            parser.error(
                "\nUnknown character '%s' in the method string is found." % primo
            )
    return methods


def checkUnitsAndMore(units):

    kB = 1.3806488 * 6.02214129 / 1000.0  # Boltzmann's constant (kJ/mol/K).
    beta = 1.0 / (kB * P.temperature)

    b_kcal = (numpy.array(['Sire', 'Amber']) == P.software.title()).any()

    if units == 'kJ':
        beta_report = beta / 4.184**b_kcal
        units = '(kJ/mol)'
    elif units == 'kcal':
        beta_report = 4.184 ** (not b_kcal) * beta
        units = '(kcal/mol)'
    elif units == 'kBT':
        beta_report = 1
        units = '(k_BT)'
    else:
        parser.error(
            '\nI don\'t understand the unit type \'%s\': the only options \'kJ\', \'kcal\', and \'kBT\''
            % units
        )

    if not P.output_directory:
        P.output_directory = P.datafile_directory
    if P.overlap:
        if not 'MBAR' in P.methods:
            parser.error("\nMBAR is not in 'methods'; can't plot the overlap matrix.")
    return units, beta, beta_report


def timeStatistics(stime):
    etime = ttt_time.time()
    tm = int((etime - stime) / 60.0)
    th = int(tm / 60.0)
    ts = '%.2f' % (etime - stime - 60 * (tm + 60 * th))
    return th, tm, ts, ttt_time.asctime()


# ===================================================================================================
# FUNCTIONS: The autocorrelation analysis.
# ===================================================================================================


def uncorrelate(sta, fin, do_dhdl=False):
    """Identifies uncorrelated samples and updates the arrays of the reduced potential energy and dhdlt retaining data entries of these samples only.
    'sta' and 'fin' are the starting and final snapshot positions to be read, both are arrays of dimension K.
    """
    print(f"🔍 uncorrelate called: K={K}, sta={sta}, fin={fin}, do_dhdl={do_dhdl}")
    print(f"🔍 nsnapshots = {fin}")
    print(f"🔍 P.uncorr_threshold = {P.uncorr_threshold}")
    if not P.uncorr_threshold:
        if P.software.title() == 'Sire':
            return dhdlt, nsnapshots, None
        return dhdlt, nsnapshots, u_klt

    u_kln = numpy.zeros(
        [K, K, max(fin - sta)], numpy.float64
    )  # u_kln[k,m,n] is the reduced potential energy of uncorrelated sample index n from state k evaluated at state m
    N_k = numpy.zeros(
        K, int
    )  # N_k[k] is the number of uncorrelated samples from state k
    g = numpy.zeros(K, float)  # autocorrelation times for the data
    if do_dhdl:
        dhdl = numpy.zeros(
            [K, n_components, max(fin - sta)], float
        )  # dhdl is value for dhdl for each component in the file at each time.
        # print("\n\nNumber of correlated and uncorrelated samples:\n\n%6s %12s %12s %12s\n") % ('State', 'N', 'N_k', 'N/N_k')
        print(
            "\n\nNumber of correlated and uncorrelated samples:\n\n"
            "%6s %12s %12s %12s\n" % ('State', 'N', 'N_k', 'N/N_k')
        )

    UNCORR_OBSERVABLE = {
        'Gromacs': P.uncorr,
        'Amber': 'dhdl',
        'Sire': 'dhdl',
        'Desmond': 'dE',
    }[P.software.title()]

    if UNCORR_OBSERVABLE == 'dhdl':
        # Uncorrelate based on dhdl values at a given lambda.

        for k in range(K):
            # Sum up over those energy components that are changing.
            # if there are repeats, we need to use the lchange[k] from the last repeated state.
            lastl = k
            for l in range(K):
                if numpy.array_equal(lv[k], lv[l]):
                    lastl = l
            dhdl_sum = numpy.sum(dhdlt[k, lchange[lastl], sta[k] : fin[k]], axis=0)
            # Determine indices of uncorrelated samples from potential autocorrelation analysis at state k

            # NML: Set statistical inefficiency (g) = 1 if vector is all 0
            if not numpy.any(dhdl_sum):
                # print "WARNING: Found all zeros for Lambda={}\n Setting statistical inefficiency g=1.".format(k)
                g[k] = 1
            else:
                # (alternatively, could use the energy differences -- here, we will use total dhdl).
                g[k] = pymbar.timeseries.statisticalInefficiency(dhdl_sum)

            indices = sta[k] + numpy.array(
                pymbar.timeseries.subsampleCorrelatedData(dhdl_sum, g=g[k])
            )  # indices of uncorrelated samples
            N_uncorr = len(indices)  # number of uncorrelated samples
            # Handle case where we end up with too few.
            if N_uncorr < P.uncorr_threshold:
                if do_dhdl:
                    # print("WARNING: Only %s uncorrelated samples found at lambda number %s; proceeding with analysis using correlated samples...") % (N_uncorr, k)
                    print(
                        "WARNING: Only %s uncorrelated samples found at lambda number %s; "
                        "proceeding with analysis using correlated samples..."
                        % (N_uncorr, k)
                    )
                indices = sta[k] + numpy.arange(len(dhdl_sum))
                N = len(indices)
            else:
                N = N_uncorr
            N_k[k] = N  # Store the number of uncorrelated samples from state k.
            if not (u_klt is None):
                for l in range(K):
                    u_kln[k, l, 0:N] = u_klt[k, l, indices]
            if do_dhdl:
                print("%6s %12s %12s %12.2f" % (k, N_uncorr, N_k[k], g[k]))
                for n in range(n_components):
                    dhdl[k, n, 0:N] = dhdlt[k, n, indices]

    if UNCORR_OBSERVABLE == 'dhdl_all':
        # Uncorrelate based on dhdl values at a given lambda.

        for k in range(K):
            # Sum up over the energy components; notice, that only the relevant data is being used in the third dimension.
            dhdl_sum = numpy.sum(dhdlt[k, :, sta[k] : fin[k]], axis=0)
            # Determine indices of uncorrelated samples from potential autocorrelation analysis at state k
            # (alternatively, could use the energy differences -- here, we will use total dhdl).
            g[k] = pymbar.timeseries.statisticalInefficiency(dhdl_sum)
            indices = sta[k] + numpy.array(
                pymbar.timeseries.subsampleCorrelatedData(dhdl_sum, g=g[k])
            )  # indices of uncorrelated samples
            N = len(indices)  # number of uncorrelated samples
            # Handle case where we end up with too few.
            if N < P.uncorr_threshold:
                if do_dhdl:
                    print(
                        "WARNING: Only %s uncorrelated samples found at lambda number %s; proceeding with analysis using correlated samples..."
                    ) % (N, k)
                indices = sta[k] + numpy.arange(len(dhdl_sum))
                N = len(indices)
            N_k[k] = N  # Store the number of uncorrelated samples from state k.
            if not (u_klt is None):
                for l in range(K):
                    u_kln[k, l, 0:N] = u_klt[k, l, indices]
            if do_dhdl:
                # print("%6s %12s %12s %12.2f") % (k, fin[k], N_k[k], g[k])
                print(f"{k:6d} {fin[k]:12d} {N_k[k]:12d} {g[k]:12.2f}")
                for n in range(n_components):
                    dhdl[k, n, 0:N] = dhdlt[k, n, indices]

    if UNCORR_OBSERVABLE == 'dE':
        print(f"🔍🔍🔍 Using dE observable")
        # Uncorrelate based on energy differences between lambdas.
        import numpy as np

        print(f"u_klt is None: {u_klt is None}")
        if u_klt is not None:
            print(f"u_klt shape: {u_klt.shape}")
            print(f"u_klt has data: {np.any(u_klt)}")

        for k in range(K):
            # Sum up over the energy components as above using only the relevant data; here we use energy differences
            # Determine indices of uncorrelated samples from potential autocorrelation analysis at state k

            dE = (
                u_klt[k, k + 1, sta[k] : fin[k]]
                if not k == K - 1
                else u_klt[k, k - 1, sta[k] : fin[k]]
            )

            print(f"🔍 State {k}: dE shape = {dE.shape}")
            print(f"🔍 State {k}: dE has data = {numpy.any(dE)}")
            print(
                f"🔍 State {k}: dE stats = min:{dE.min():.3f}, max:{dE.max():.3f}, std:{dE.std():.3f}"
            )
            if not numpy.any(dE):
                print(f"🔍 State {k}: dE is all zeros!")

            g[k] = pymbar.timeseries.statisticalInefficiency(dE)
            indices = sta[k] + numpy.array(
                pymbar.timeseries.subsampleCorrelatedData(dE, g=g[k])
            )  # indices of uncorrelated samples
            N = len(indices)  # number of uncorrelated samples
            # Handle case where we end up with too few.
            if N < P.uncorr_threshold:
                print(
                    "WARNING: Only %s uncorrelated samples found at lambda number %s; proceeding with analysis using correlated samples..."
                ) % (N, k)
                indices = sta[k] + numpy.arange(len(dE))
                N = len(indices)
            N_k[k] = N  # Store the number of uncorrelated samples from state k.
            if not (u_klt is None):
                for l in range(K):
                    u_kln[k, l, 0:N] = u_klt[k, l, indices]
                    # ADD THIS PRINT STATEMENT:
            if True:  # This should probably just be True for dE
                print("-----%6s %12s %12s %12.2f" % (k, len(indices), N_k[k], g[k]))

    if do_dhdl:
        return (dhdl, N_k, u_kln)
    return (N_k, u_kln)


# ===================================================================================================
# FUNCTIONS: The MBAR workhorse.
# ===================================================================================================


def estimatewithMBAR(u_kln, N_k, reltol, regular_estimate=False):
    """Computes the MBAR free energy given the reduced potential and the number of relevant entries in it."""

    def plotOverlapMatrix(O):
        """Plots the probability of observing a sample from state i (row) in state j (column).
        For convenience, the neigboring state cells are fringed in bold."""
        max_prob = O.max()
        fig = pl.figure(figsize=(K / 2.0, K / 2.0))
        fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

        for i in range(K):
            if i != 0:
                pl.axvline(x=i, ls='-', lw=0.5, color='k', alpha=0.25)
                pl.axhline(y=i, ls='-', lw=0.5, color='k', alpha=0.25)
            for j in range(K):
                if O[j, i] < 0.005:
                    ii = ''
                elif O[j, i] > 0.995:
                    ii = '1.00'
                else:
                    ii = ("%.2f" % O[j, i])[1:]
                alf = O[j, i] / max_prob
                pl.fill_between(
                    [i, i + 1],
                    [K - j, K - j],
                    [K - (j + 1), K - (j + 1)],
                    color='k',
                    alpha=alf,
                )
                pl.annotate(
                    ii,
                    xy=(i, j),
                    xytext=(i + 0.5, K - (j + 0.5)),
                    size=8,
                    textcoords='data',
                    va='center',
                    ha='center',
                    color=('k' if alf < 0.5 else 'w'),
                )

        if P.bSkipLambdaIndex:
            ks = [int(l) for l in P.bSkipLambdaIndex.split('-')]
            ks = numpy.delete(numpy.arange(K + len(ks)), ks)
        else:
            ks = range(K)
        for i in range(K):
            pl.annotate(
                ks[i],
                xy=(i + 0.5, 1),
                xytext=(i + 0.5, K + 0.5),
                size=10,
                textcoords=('data', 'data'),
                va='center',
                ha='center',
                color='k',
            )
            pl.annotate(
                ks[i],
                xy=(-0.5, K - (j + 0.5)),
                xytext=(-0.5, K - (i + 0.5)),
                size=10,
                textcoords=('data', 'data'),
                va='center',
                ha='center',
                color='k',
            )
        pl.annotate(
            '$\lambda$',
            xy=(-0.5, K - (j + 0.5)),
            xytext=(-0.5, K + 0.5),
            size=10,
            textcoords=('data', 'data'),
            va='center',
            ha='center',
            color='k',
        )
        pl.plot([0, K], [0, 0], 'k-', lw=4.0, solid_capstyle='butt')
        pl.plot([K, K], [0, K], 'k-', lw=4.0, solid_capstyle='butt')
        pl.plot([0, 0], [0, K], 'k-', lw=2.0, solid_capstyle='butt')
        pl.plot([0, K], [K, K], 'k-', lw=2.0, solid_capstyle='butt')

        # cx = sorted(2*range(K+1))
        # cy = sorted(2*range(K+1), reverse=True)
        cx = sorted([2 * i for i in range(K + 1)])
        cy = sorted([2 * i for i in range(K + 1)], reverse=True)
        pl.plot(cx[2:-1], cy[1:-2], 'k-', lw=2.0)
        pl.plot(numpy.array(cx[2:-3]) + 1, cy[1:-4], 'k-', lw=2.0)
        pl.plot(cx[1:-2], numpy.array(cy[:-3]) - 1, 'k-', lw=2.0)
        pl.plot(cx[1:-4], numpy.array(cy[:-5]) - 2, 'k-', lw=2.0)

        pl.xlim(-1, K)
        pl.ylim(0, K + 1)
        pl.savefig(
            os.path.join(P.output_directory, 'O_MBAR.pdf'),
            bbox_inches='tight',
            pad_inches=0.0,
        )
        pl.close(fig)
        return

    if regular_estimate:
        print("\nEstimating the free energy change with MBAR...")
    MBAR = pymbar.mbar.MBAR(
        u_kln, N_k, verbose=P.verbose, relative_tolerance=reltol, initialize=P.init_with
    )
    # Get matrix of dimensionless free energy differences and uncertainty estimate.
    (Deltaf_ij, dDeltaf_ij, theta_ij) = MBAR.getFreeEnergyDifferences(
        uncertainty_method='svd-ew', return_theta=True
    )
    if P.verbose:
        print(
            f"Matrix of free energy differences\n"
            f"Deltaf_ij:\n{Deltaf_ij}\n"
            f"dDeltaf_ij:\n{dDeltaf_ij}"
        )
    if regular_estimate:
        if P.overlap:
            print("The overlap matrix is...")
            ov = MBAR.computeOverlap()
            if isinstance(ov, dict):
                O = ov['matrix']
            else:
                # older versions
                O = ov[2]
            for k in range(K):
                line = ''
                for l in range(K):
                    line += ' %5.2f ' % O[k, l]
                print(line)
            plotOverlapMatrix(O)
            print("\nFor a nicer figure look at 'O_MBAR.pdf'")
        return (Deltaf_ij, dDeltaf_ij)
    return (Deltaf_ij[0, K - 1] / P.beta_report, dDeltaf_ij[0, K - 1] / P.beta_report)


# ===================================================================================================
# FUNCTIONS: Thermodynamic integration.
# ===================================================================================================


class naturalcubicspline:

    def __init__(self, x):

        # define some space
        L = len(x)
        H = numpy.zeros([L, L], float)
        M = numpy.zeros([L, L], float)
        BW = numpy.zeros([L, L], float)
        AW = numpy.zeros([L, L], float)
        DW = numpy.zeros([L, L], float)

        h = x[1:L] - x[0 : L - 1]
        ih = 1.0 / h

        # define the H and M matrix, from p. 371 "applied numerical methods with matlab, Chapra"
        H[0, 0] = 1
        H[L - 1, L - 1] = 1
        for i in range(1, L - 1):
            H[i, i] = 2 * (h[i - 1] + h[i])
            H[i, i - 1] = h[i - 1]
            H[i, i + 1] = h[i]

            M[i, i] = -3 * (ih[i - 1] + ih[i])
            M[i, i - 1] = 3 * (ih[i - 1])
            M[i, i + 1] = 3 * (ih[i])

        CW = numpy.dot(
            numpy.linalg.inv(H), M
        )  # this is the matrix translating c to weights in f.
        # each row corresponds to the weights for each c.

        # from CW, define the other coefficient matrices
        for i in range(0, L - 1):
            BW[i, :] = -(h[i] / 3) * (2 * CW[i, :] + CW[i + 1, :])
            BW[i, i] += -ih[i]
            BW[i, i + 1] += ih[i]
            DW[i, :] = (ih[i] / 3) * (CW[i + 1, :] - CW[i, :])
            AW[i, i] = 1

        # Make copies of the arrays we'll be using in the future.
        self.x = x.copy()
        self.AW = AW.copy()
        self.BW = BW.copy()
        self.CW = CW.copy()
        self.DW = DW.copy()

        # find the integrating weights
        self.wsum = numpy.zeros([L], float)
        self.wk = numpy.zeros([L - 1, L], float)
        for k in range(0, L - 1):
            w = (
                DW[k, :] * (h[k] ** 4) / 4.0
                + CW[k, :] * (h[k] ** 3) / 3.0
                + BW[k, :] * (h[k] ** 2) / 2.0
                + AW[k, :] * (h[k])
            )
            self.wk[k, :] = w
            self.wsum += w

    def interpolate(self, y, xnew):
        if len(self.x) != len(y):
            parser.error(
                "\nThe length of 'y' should be consistent with that of 'self.x'. I cannot perform linear algebra operations."
            )
        # get the array of actual coefficients by multiplying the coefficient matrix by the values
        a = numpy.dot(self.AW, y)
        b = numpy.dot(self.BW, y)
        c = numpy.dot(self.CW, y)
        d = numpy.dot(self.DW, y)

        N = len(xnew)
        ynew = numpy.zeros([N], float)
        for i in range(N):
            # Find the index of 'xnew[i]' it would have in 'self.x'.
            j = numpy.searchsorted(self.x, xnew[i]) - 1
            lamw = xnew[i] - self.x[j]
            ynew[i] = d[j] * lamw**3 + c[j] * lamw**2 + b[j] * lamw + a[j]
        # Preserve the terminal points.
        ynew[0] = y[0]
        ynew[-1] = y[-1]
        return ynew


def get_lchange(lv):
    lchange = numpy.zeros(
        [K, n_components], bool
    )  # booleans for which lambdas are changing
    for j in range(n_components):
        # need to identify range over which lambda doesn't change, and not interpolate over that range.
        for k in range(K - 1):
            if lv[k + 1, j] - lv[k, j] > 0:
                lchange[k, j] = True
                lchange[k + 1, j] = True
    return lchange


def TIprelim(lv):

    # Lambda vectors spacing.
    dlam = numpy.diff(lv, axis=0)

    if 'ave_dhdl' in globals() and 'std_dhdl' in globals():
        return dlam, globals()['ave_dhdl'], globals()['std_dhdl']

    # Compute <dhdl> and std(dhdl) for each component, for each lambda; multiply them by beta to make unitless.
    ave_dhdl = numpy.zeros([K, n_components], float)
    std_dhdl = numpy.zeros([K, n_components], float)
    # print("check ave_dhdl and std_dhdl", ave_dhdl, std_dhdl)
    for k in range(K):
        ave_dhdl[k, :] = P.beta * numpy.average(dhdl[k, :, 0 : N_k[k]], axis=1)
        std_dhdl[k, :] = (
            P.beta * numpy.std(dhdl[k, :, 0 : N_k[k]], axis=1) / numpy.sqrt(N_k[k] - 1)
        )
    print("check ave_dhdl and std_dhdl", ave_dhdl, std_dhdl)
    return dlam, ave_dhdl, std_dhdl


def getSplines(lchange):
    # construct a map back to the original components
    mapl = numpy.zeros(
        [K, n_components], int
    )  # map back to the original k from the components
    for j in range(n_components):
        incr = 0
        for k in range(K):
            if lchange[k, j]:
                mapl[k, j] += incr
                incr += 1

    # put together the spline weights for the different components
    cubspl = list()
    for j in range(n_components):
        lv_lchange = lv[lchange[:, j], j]
        if len(lv_lchange) == 0:  # handle the all-zero lv column
            cubspl.append(0)
        else:
            spl = naturalcubicspline(lv_lchange)
            cubspl.append(spl)
    return cubspl, mapl


# ===================================================================================================
# FUNCTIONS: This one estimates dF and ddF for all pairs of adjacent states and stores them.
# ===================================================================================================


def estimatePairs():

    # print("Estimating the free energy change with %s..." % ', '.join(P.methods)).replace(', MBAR', '')
    print(
        "Estimating the free energy change with %s..."
        % ', '.join(P.methods).replace(', MBAR', '')
    )
    df_allk = list()
    ddf_allk = list()

    for k in range(K - 1):
        df = dict()
        ddf = dict()

        for name in P.methods:

            if name == 'TI':
                # ===================================================================================================
                # Estimate free energy difference with TI; interpolating with the trapezoidal rule.
                # ===================================================================================================
                df['TI'] = 0.5 * numpy.dot(dlam[k], (ave_dhdl[k] + ave_dhdl[k + 1]))
                ddf['TI'] = 0.5 * numpy.sqrt(
                    numpy.dot(dlam[k] ** 2, std_dhdl[k] ** 2 + std_dhdl[k + 1] ** 2)
                )

            if name == 'TI-CUBIC':
                # ===================================================================================================
                # Estimate free energy difference with TI; interpolating with the natural cubic splines.
                # ===================================================================================================
                df['TI-CUBIC'], ddf['TI-CUBIC'] = 0, 0
                for j in range(n_components):
                    if dlam[k, j] > 0:
                        lj = lchange[:, j]
                        df['TI-CUBIC'] += numpy.dot(
                            cubspl[j].wk[mapl[k, j]], ave_dhdl[lj, j]
                        )
                        ddf['TI-CUBIC'] += numpy.dot(
                            cubspl[j].wk[mapl[k, j]] ** 2, std_dhdl[lj, j] ** 2
                        )
                ddf['TI-CUBIC'] = numpy.sqrt(ddf['TI-CUBIC'])

            if any(name == m for m in ['DEXP', 'GDEL', 'BAR', 'UBAR', 'RBAR']):
                w_F = u_kln[k, k + 1, 0 : N_k[k]] - u_kln[k, k, 0 : N_k[k]]

            if name == 'DEXP':
                # ===================================================================================================
                # Estimate free energy difference with Forward-direction EXP (in this case, Deletion from solvent).
                # ===================================================================================================
                (df['DEXP'], ddf['DEXP']) = pymbar.exp.EXP(w_F)

            if name == 'GDEL':
                # ===================================================================================================
                # Estimate free energy difference with a Gaussian estimate of EXP (in this case, deletion from solvent)
                # ===================================================================================================
                (df['GDEL'], ddf['GDEL']) = pymbar.exp.EXPGauss(w_F)

            if any(name == m for m in ['IEXP', 'GINS', 'BAR', 'UBAR', 'RBAR']):
                w_R = (
                    u_kln[k + 1, k, 0 : N_k[k + 1]]
                    - u_kln[k + 1, k + 1, 0 : N_k[k + 1]]
                )

            if name == 'IEXP':
                # ===================================================================================================
                # Estimate free energy difference with Reverse-direction EXP (in this case, insertion into solvent).
                # ===================================================================================================
                print("check if w_R is zero", w_R)
                (rdf, rddf) = pymbar.exp.EXP(w_R)
                (df['IEXP'], ddf['IEXP']) = (-rdf, rddf)

            if name == 'GINS':
                # ===================================================================================================
                # Estimate free energy difference with a Gaussian estimate of EXP (in this case, insertion into solvent)
                # ===================================================================================================
                (rdf, rddf) = pymbar.exp.EXPGauss(w_R)
                (df['GINS'], ddf['GINS']) = (-rdf, rddf)

            if name == 'BAR':
                # ===================================================================================================
                # Estimate free energy difference with BAR; use w_F and w_R computed above.
                # ===================================================================================================
                (df['BAR'], ddf['BAR']) = pymbar.bar.BAR(
                    w_F, w_R, relative_tolerance=P.relative_tolerance, verbose=P.verbose
                )

            if name == 'UBAR':
                # ===================================================================================================
                # Estimate free energy difference with unoptimized BAR -- assume dF is zero, and just do one evaluation
                # ===================================================================================================
                (df['UBAR'], ddf['UBAR']) = pymbar.bar.BAR(
                    w_F, w_R, verbose=P.verbose, iterated_solution=False
                )

            if name == 'RBAR':
                # ===================================================================================================
                # Estimate free energy difference with Unoptimized BAR over range of free energy values, and choose the one
                # that is self consistently best.
                # ===================================================================================================
                min_diff = 1e6
                best_udf = 0
                for trial_udf in range(-10, 10, 1):
                    (udf, uddf) = pymbar.bar.BAR(
                        w_F,
                        w_R,
                        DeltaF=trial_udf,
                        iterated_solution=False,
                        verbose=P.verbose,
                    )
                    diff = numpy.abs(udf - trial_udf)
                    if diff < min_diff:
                        best_udf = udf
                        best_uddf = uddf
                        min_diff = diff
                (df['RBAR'], ddf['RBAR']) = (best_udf, best_uddf)

            if name == 'MBAR':
                # ===================================================================================================
                # Store the MBAR free energy difference (already estimated above) properly, i.e. by state.
                # ===================================================================================================
                (df['MBAR'], ddf['MBAR']) = Deltaf_ij[k, k + 1], numpy.nan_to_num(
                    dDeltaf_ij[k, k + 1]
                )

        df_allk = numpy.append(df_allk, df)
        ddf_allk = numpy.append(ddf_allk, ddf)

        # Save full-precision data to pickle
        pairs_data = {'df_allk': df_allk, 'ddf_allk': ddf_allk}
        pickle_path = os.path.join('./', 'pairs_results.pickle')
        with open(pickle_path, 'wb') as f:
            pickle.dump(pairs_data, f)
        print(f"Saved pairwise free energies to {pickle_path}")

    return df_allk, ddf_allk


# ===================================================================================================
# FUNCTIONS: All done with calculations; summarize and print stats.
# ===================================================================================================


def totalEnergies():

    # Count up the charging states.
    startcoul = 0
    endcoul = 0
    startvdw = 0
    endvdw = 0
    for lv_n in ['vdw', 'coul', 'fep']:
        if lv_n in P.lv_names:
            _ndx_char = P.lv_names.index(lv_n)
            lv_char = lv[:, _ndx_char]
            if not (lv_char == lv_char[0]).all():
                if lv_n == 'vdw':
                    endvdw = (lv_char != 1).sum()
                if lv_n == 'fep':
                    endcoul = (lv_char != 1).sum()  # Why is this lumped with coul?
                    ndx_char = P.lv_names.index(lv_n)
                if lv_n == 'coul':
                    endcoul = (lv_char != 1).sum()
                    ndx_char = P.lv_names.index(lv_n)
    # Figure out where Coulomb section is, if it is present.
    if endcoul > endvdw:
        # If it comes after the vdW section, for now assume it follows vdW (will need correcting for case where they are mixed together)
        startcoul = endvdw
        startvdw = 0
    elif startcoul == endcoul:
        # There is no coulomb section
        if P.verbose:
            print("No Coulomb transformation present.")
        pass
    else:
        startcoul = 0
        startvdw = endcoul

    segments = ['Coulomb', 'vdWaals', 'TOTAL']
    segmentstarts = [startcoul, startvdw, 0]
    segmentends = [min(endcoul, K - 1), min(endvdw, K - 1), K - 1]
    dFs = []
    ddFs = []

    # Perform the energy segmentation; be pedantic about the TI cumulative ddF's (see Section 3.1 of the paper).
    for i in range(len(segments)):
        segment = segments[i]
        segstart = segmentstarts[i]
        segend = segmentends[i]
        dF = dict.fromkeys(P.methods, 0)
        ddF = dict.fromkeys(P.methods, 0)

        for name in P.methods:
            if name[0:2] == 'TI':
                for k in range(segstart, segend):
                    # print("check df_allk[k][name]", df_allk[k][name], k, name)
                    dF[name] += df_allk[k][name]

                if segment == 'Coulomb':
                    jlist = [ndx_char] if endcoul > 0 else []
                elif segment == 'vdWaals':
                    jlist = []
                elif segment == 'TOTAL':
                    jlist = range(n_components)

                for j in jlist:
                    lj = lchange[:, j]
                    if not (lj == False).all():  # handle the all-zero lv column
                        if name == 'TI-CUBIC':
                            ddF[name] += numpy.dot(
                                (cubspl[j].wsum) ** 2, std_dhdl[lj, j] ** 2
                            )
                        elif name == 'TI':
                            h = numpy.trim_zeros(dlam[:, j])
                            wsum = 0.5 * (numpy.append(h, 0) + numpy.append(0, h))
                            ddF[name] += numpy.dot(wsum**2, std_dhdl[lj, j] ** 2)

                ddF[name] = numpy.sqrt(ddF[name])

            # Use the total energy value and uncertainty that pymbar offers.
            elif name == 'MBAR':
                dF[name] = Deltaf_ij[segstart, segend]
                ddF[name] = dDeltaf_ij[segstart, segend]

            else:
                for k in range(segstart, segend):
                    dF[name] += df_allk[k][name]
                    ddF[name] += (ddf_allk[k][name]) ** 2
                ddF[name] = numpy.sqrt(ddF[name])

        dFs.append(dF)
        ddFs.append(ddF)

    for name in P.methods:  # 'vdWaals' = 'TOTAL' - 'Coulomb'
        ddFs[1][name] = (ddFs[2][name] ** 2 - ddFs[0][name] ** 2) ** 0.5

    # Display results.
    def printLine(str1, str2, d1=None, d2=None):
        """Fills out the results table linewise."""
        print(str1),
        text = str1
        for name in P.methods:
            if d1 == 'plain':
                print(str2),
                text += ' ' + str2
            if d1 == 'name':
                print(str2, (name, P.units)),
                text += ' ' + str2 % (name, P.units)
            if d1 and d2:
                print(str2, (d1[name] / P.beta_report, d2[name] / P.beta_report)),
                text += ' ' + str2 % (
                    d1[name] / P.beta_report,
                    d2[name] / P.beta_report,
                )
        print('')
        outtext.append(text + '\n')
        return

    d = P.decimal
    str_dash = (d + 7 + 6 + d + 2) * '-'
    str_dat = ('X%d.%df  +-  X%d.%df' % (d + 7, d, d + 2, d)).replace('X', '%')
    str_names = ('X%ds X-%ds' % (d + 6, d + 8)).replace('X', '%')
    outtext = []
    printLine(12 * '-', str_dash, 'plain')
    printLine('%-12s' % '   States', str_names, 'name')
    printLine(12 * '-', str_dash, 'plain')
    for k in range(K - 1):
        printLine('%4d -- %-4d' % (k, k + 1), str_dat, df_allk[k], ddf_allk[k])
    printLine(12 * '-', str_dash, 'plain')
    remark = [
        "",
        "A remark on the energy components interpretation: ",
        " 'vdWaals' is computed as 'TOTAL' - 'Coulomb', where ",
        " 'Coulomb' is found as the free energy change between ",
        " the states defined by the lambda vectors (0,0,...,0) ",
        " and (1,0,...,0), the only varying vector component ",
        " being either 'coul-lambda' or 'fep-lambda'. ",
    ]
    w = 12 + (1 + len(str_dash)) * len(P.methods)
    str_align = '{:I^%d}' % w
    if len(P.lv_names) > 1:
        for i in range(len(segments)):
            printLine('%9s:  ' % segments[i], str_dat, dFs[i], ddFs[i])
        for i in remark:
            print(str_align.replace('I', ' ').format(i))
    else:
        printLine('%9s:  ' % segments[-1], str_dat, dFs[-1], ddFs[-1])
    # Store results.
    outfile = open(os.path.join(P.output_directory, '%s.txt' % P.resultfilename), 'w')
    outfile.write('# Command line was: %s\n\n' % ' '.join(sys.argv))
    outfile.writelines(outtext)
    outfile.close()

    P.datafile_directory = os.getcwd()
    P.when_analyzed = ttt_time.asctime()
    P.ddf_allk = ddf_allk
    P.df_allk = df_allk
    P.ddFs = ddFs
    P.dFs = dFs

    outfile = open(
        os.path.join(P.output_directory, '%s.pickle' % P.resultfilename), 'wb'
    )
    pickle.dump(P, outfile)
    outfile.close()

    print('\n' + w * '*')
    for i in [
        " The above table has been stored in ",
        " " + P.output_directory + "/%s.txt " % P.resultfilename,
        " while the full-precision data ",
        " (along with the simulation profile) in ",
        " " + P.output_directory + "/%s.pickle " % P.resultfilename,
    ]:
        print(str_align.format('{:^40}'.format(i)))
    print(w * '*')

    return


# ===================================================================================================
# FUNCTIONS: Free energy change vs. simulation time. Called by the -f flag.
# ===================================================================================================


def dF_t():

    def plotdFvsTime(f_ts, r_ts, F_df, R_df, F_ddf, R_ddf):
        """Plots the free energy change computed using the equilibrated snapshots between the proper target time frames (f_ts and r_ts)
        in both forward (data points are stored in F_df and F_ddf) and reverse (data points are stored in R_df and R_ddf) directions.
        """
        fig = pl.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        pl.setp(ax.spines['bottom'], color='#D2B9D3', lw=3, zorder=-2)
        pl.setp(ax.spines['left'], color='#D2B9D3', lw=3, zorder=-2)
        for dire in ['top', 'right']:
            ax.spines[dire].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        max_fts = max(f_ts)
        rr_ts = [aa / max_fts for aa in f_ts[::-1]]
        f_ts = [aa / max_fts for aa in f_ts]
        r_ts = [aa / max_fts for aa in r_ts]

        line0 = pl.fill_between(
            [r_ts[0], f_ts[-1]],
            R_df[0] - R_ddf[0],
            R_df[0] + R_ddf[0],
            color='#D2B9D3',
            zorder=-5,
        )
        for i in range(len(f_ts)):
            line1 = pl.plot(
                [f_ts[i]] * 2,
                [F_df[i] - F_ddf[i], F_df[i] + F_ddf[i]],
                color='#736AFF',
                ls='-',
                lw=3,
                solid_capstyle='round',
                zorder=1,
            )
        line11 = pl.plot(
            f_ts,
            F_df,
            color='#736AFF',
            ls='-',
            lw=3,
            marker='o',
            mfc='w',
            mew=2.5,
            mec='#736AFF',
            ms=12,
            zorder=2,
        )

        for i in range(len(rr_ts)):
            line2 = pl.plot(
                [rr_ts[i]] * 2,
                [R_df[i] - R_ddf[i], R_df[i] + R_ddf[i]],
                color='#C11B17',
                ls='-',
                lw=3,
                solid_capstyle='round',
                zorder=3,
            )
        line22 = pl.plot(
            rr_ts,
            R_df,
            color='#C11B17',
            ls='-',
            lw=3,
            marker='o',
            mfc='w',
            mew=2.5,
            mec='#C11B17',
            ms=12,
            zorder=4,
        )

        pl.xlim(r_ts[0], f_ts[-1])

        pl.xticks(r_ts[::2] + f_ts[-1:], fontsize=10)
        pl.yticks(fontsize=10)

        leg = pl.legend(
            (line1[0], line2[0]),
            (r'$Forward$', r'$Reverse$'),
            loc=9,
            prop=FP(size=18),
            frameon=False,
        )
        pl.xlabel(
            r'$\mathrm{Fraction\/of\/the\/simulation\/time}$',
            fontsize=16,
            color='#151B54',
        )
        pl.ylabel(r'$\mathrm{\Delta G\/%s}$' % P.units, fontsize=16, color='#151B54')
        pl.xticks(f_ts, ['%.2f' % i for i in f_ts])
        pl.tick_params(axis='x', color='#D2B9D3')
        pl.tick_params(axis='y', color='#D2B9D3')
        pl.savefig(os.path.join(P.output_directory, 'dF_t.pdf'))
        pl.close(fig)
        return

    if not 'MBAR' in P.methods:
        parser.error(
            "\nCurrent version of the dF(t) analysis works with MBAR only and the method is not found in the list."
        )
    if not (
        P.snap_size[0] == numpy.array(P.snap_size)
    ).all():  # this could be circumvented
        parser.error(
            "\nThe snapshot size isn't the same for all the files; cannot perform the dF(t) analysis."
        )

    # Define a list of bForwrev equidistant time frames at which the free energy is to be estimated; count up the snapshots embounded between the time frames.
    n_tf = P.bForwrev + 1
    nss_tf = numpy.zeros([n_tf, K], int)
    increment = 1.0 / (n_tf - 1)
    if P.bExpanded:
        from collections import Counter  # for counting elements in an array

        # tf = numpy.arange(0,1+increment,increment)*(numpy.sum(nsnapshots)-1)+1
        tf = numpy.linspace(0.0, 1.0, n_tf) * (numpy.sum(nsnapshots) - 1) + 1
        tf[0] = 0
        for i in range(n_tf - 1):
            nss = Counter(extract_states[tf[i] : tf[i + 1]])
            nss_tf[i + 1] = numpy.array([nss[j] for j in range(K)])
    else:
        # print "Increment:", increment
        # print "Max snapshots:", max(nsnapshots)
        # tf = numpy.arange(0,1+increment,increment)*(max(nsnapshots)-1)+1
        tf = numpy.linspace(0.0, 1.0, n_tf) * (max(nsnapshots) - 1) + 1
        # print tf, tf_new
        # print len(tf), len(tf_new)
        # print("Pausing."), raw_input()
        tf[0] = 0
        for i in range(n_tf - 1):
            nss_tf[i + 1] = numpy.array(
                [min(j, tf[i + 1]) for j in nsnapshots]
            ) - numpy.sum(nss_tf[: i + 1], axis=0)

    # Define the real time scale (in ps) rather than a snapshot sequence.
    ts = ["%.1f" % ((i - (i != tf[0])) * P.snap_size[0] + P.equiltime) for i in tf]
    # Initialize arrays to store data points to be plotted.
    F_df = numpy.zeros(n_tf - 1, float)
    F_ddf = numpy.zeros(n_tf - 1, float)
    R_df = numpy.zeros(n_tf - 1, float)
    R_ddf = numpy.zeros(n_tf - 1, float)
    # Store the MBAR energy that accounts for all the equilibrated snapshots (has already been computed in the previous section).
    F_df[-1], F_ddf[-1] = (
        Deltaf_ij[0, K - 1] / P.beta_report,
        dDeltaf_ij[0, K - 1] / P.beta_report,
    )
    R_df[0], R_ddf[0] = (
        Deltaf_ij[0, K - 1] / P.beta_report,
        dDeltaf_ij[0, K - 1] / P.beta_report,
    )
    # Do the forward analysis.
    print(
        "Forward dF(t) analysis...\nEstimating the free energy change using the data up to"
    )
    sta = nss_tf[0]
    for i in range(n_tf - 2):
        print("%60s ps...") % ts[i + 1]
        fin = numpy.sum(nss_tf[: i + 2], axis=0)
        N_k, u_kln = uncorrelate(nss_tf[0], numpy.sum(nss_tf[: i + 2], axis=0))
        F_df[i], F_ddf[i] = estimatewithMBAR(u_kln, N_k, P.relative_tolerance)
    # Do the reverse analysis.
    print("Reverse dF(t) analysis...\nUsing the data starting from")
    fin = numpy.sum(nss_tf[:], axis=0)
    for i in range(n_tf - 2):
        print("%34s ps...") % ts[i + 1]
        sta = numpy.sum(nss_tf[: i + 2], axis=0)
        N_k, u_kln = uncorrelate(sta, fin)
        R_df[i + 1], R_ddf[i + 1] = estimatewithMBAR(u_kln, N_k, P.relative_tolerance)

    print(
        """\n   The free energies %s evaluated by using the trajectory
   snaphots corresponding to various time intervals for both the
   reverse and forward (in parentheses) direction.\n"""
    ) % P.units
    print("%s\n %20s %19s %20s\n%s") % (
        70 * '-',
        'Time interval, ps',
        'Reverse',
        'Forward',
        70 * '-',
    )
    print("%10s -- %s\n%10s -- %-10s %11.3f +- %5.3f %16s\n") % (
        ts[0],
        ts[-1],
        '(' + ts[0],
        ts[0] + ')',
        R_df[0],
        R_ddf[0],
        'XXXXXX',
    )
    for i in range(1, len(ts) - 1):
        print("%10s -- %s\n%10s -- %-10s %11.3f +- %5.3f %11.3f +- %5.3f\n") % (
            ts[i],
            ts[-1],
            '(' + ts[0],
            ts[i] + ')',
            R_df[i],
            R_ddf[i],
            F_df[i - 1],
            F_ddf[i - 1],
        )
    print("%10s -- %s\n%10s -- %-10s %16s %15.3f +- %5.3f\n%s") % (
        ts[-1],
        ts[-1],
        '(' + ts[0],
        ts[-1] + ')',
        'XXXXXX',
        F_df[-1],
        F_ddf[-1],
        70 * '-',
    )

    # Plot the forward and reverse dF(t); store the data points in the text file.
    print("Plotting data to the file dF_t.pdf...\n\n")
    plotdFvsTime(
        [float(i) for i in ts[1:]],
        [float(i) for i in ts[:-1]],
        F_df,
        R_df,
        F_ddf,
        R_ddf,
    )
    outtext = [
        "%12s %10s %-10s %17s %10s %s\n"
        % ('Time (ps)', 'Forward', P.units, 'Time (ps)', 'Reverse', P.units)
    ]
    outtext += [
        "%10s %11.3f +- %5.3f %18s %11.3f +- %5.3f\n"
        % (ts[1:][i], F_df[i], F_ddf[i], ts[:-1][i], R_df[i], R_ddf[i])
        for i in range(len(F_df))
    ]
    outfile = open(os.path.join(P.output_directory, 'dF_t.txt'), 'w')
    outfile.writelines(outtext)
    outfile.close()
    return


# ===================================================================================================
# FUNCTIONS: Free energy change breakdown (into lambda-pair dFs). Called by the -g flag.
# ===================================================================================================


def plotdFvsLambda():

    def plotdFvsLambda1():
        """Plots the free energy differences evaluated for each pair of adjacent states for all methods."""
        x = numpy.arange(len(df_allk))
        if x[-1] < 8:
            fig = pl.figure(figsize=(8, 6))
        else:
            fig = pl.figure(figsize=(len(x), 6))
        width = 1.0 / (len(P.methods) + 1)
        elw = 30 * width
        colors = {
            'TI': '#C45AEC',
            'TI-CUBIC': '#33CC33',
            'DEXP': '#F87431',
            'IEXP': '#FF3030',
            'GINS': '#EAC117',
            'GDEL': '#347235',
            'BAR': '#6698FF',
            'UBAR': '#817339',
            'RBAR': '#C11B17',
            'MBAR': '#F9B7FF',
        }
        lines = tuple()
        for name in P.methods:
            y = [df_allk[i][name] / P.beta_report for i in x]
            ye = [ddf_allk[i][name] / P.beta_report for i in x]
            line = pl.bar(
                x + len(lines) * width,
                y,
                width,
                color=colors[name],
                yerr=ye,
                lw=0.1 * elw,
                error_kw=dict(elinewidth=elw, ecolor='black', capsize=0.5 * elw),
            )
            lines += (line[0],)
        pl.xlabel('States', fontsize=12, color='#151B54')
        pl.ylabel('$\Delta G$ ' + P.units, fontsize=12, color='#151B54')
        pl.xticks(
            x + 0.5 * width * len(P.methods),
            tuple(['%d--%d' % (i, i + 1) for i in x]),
            fontsize=8,
        )
        pl.yticks(fontsize=8)
        pl.xlim(x[0], x[-1] + len(lines) * width)
        ax = pl.gca()
        for dir in ['right', 'top', 'bottom']:
            ax.spines[dir].set_color('none')
        ax.yaxis.set_ticks_position('left')
        for tick in ax.get_xticklines():
            tick.set_visible(False)

        leg = pl.legend(
            lines, tuple(P.methods), loc=3, ncol=2, prop=FP(size=10), fancybox=True
        )
        leg.get_frame().set_alpha(0.5)
        pl.title('The free energy change breakdown', fontsize=12)
        pl.savefig(
            os.path.join(P.output_directory, 'dF_state_long.pdf'), bbox_inches='tight'
        )
        pl.close(fig)
        return

    def plotdFvsLambda2(nb=10):
        """Plots the free energy differences evaluated for each pair of adjacent states for all methods.
        The layout is approximately 'nb' bars per subplot."""
        x = numpy.arange(len(df_allk))
        if len(x) < nb:
            return
        xs = numpy.array_split(x, len(x) / nb + 1)
        mnb = max([len(i) for i in xs])
        fig = pl.figure(figsize=(8, 6))
        width = 1.0 / (len(P.methods) + 1)
        elw = 30 * width
        colors = {
            'TI': '#C45AEC',
            'TI-CUBIC': '#33CC33',
            'DEXP': '#F87431',
            'IEXP': '#FF3030',
            'GINS': '#EAC117',
            'GDEL': '#347235',
            'BAR': '#6698FF',
            'UBAR': '#817339',
            'RBAR': '#C11B17',
            'MBAR': '#F9B7FF',
        }
        ndx = 1
        for x in xs:
            lines = tuple()
            ax = pl.subplot(len(xs), 1, ndx)
            for name in P.methods:
                y = [df_allk[i][name] / P.beta_report for i in x]
                ye = [ddf_allk[i][name] / P.beta_report for i in x]
                line = pl.bar(
                    x + len(lines) * width,
                    y,
                    width,
                    color=colors[name],
                    yerr=ye,
                    lw=0.05 * elw,
                    error_kw=dict(elinewidth=elw, ecolor='black', capsize=0.5 * elw),
                )
                lines += (line[0],)
            for dir in ['left', 'right', 'top', 'bottom']:
                if dir == 'left':
                    ax.yaxis.set_ticks_position(dir)
                else:
                    ax.spines[dir].set_color('none')
            pl.yticks(fontsize=10)
            ax.xaxis.set_ticks([])
            for i in x + 0.5 * width * len(P.methods):
                ax.annotate(
                    '$\mathrm{%d-%d}$' % (i, i + 1),
                    xy=(i, 0),
                    xycoords=('data', 'axes fraction'),
                    xytext=(0, -2),
                    size=10,
                    textcoords='offset points',
                    va='top',
                    ha='center',
                )
            pl.xlim(x[0], x[-1] + len(lines) * width + (mnb - len(x)))
            ndx += 1
        leg = ax.legend(
            lines,
            tuple(P.methods),
            loc=0,
            ncol=2,
            prop=FP(size=8),
            title='$\mathrm{\Delta G\/%s\/}\mathit{vs.}\/\mathrm{lambda\/pair}$'
            % P.units,
            fancybox=True,
        )
        leg.get_frame().set_alpha(0.5)
        pl.savefig(
            os.path.join(P.output_directory, 'dF_state.pdf'), bbox_inches='tight'
        )
        pl.close(fig)
        return

    def plotTI():
        """Plots the ave_dhdl array as a function of the lambda value.
        If (TI and TI-CUBIC in methods) -- plots the TI integration area and the TI-CUBIC interpolation curve,
        elif (only one of them in methods) -- plots the integration area of the method.
        """
        min_dl = dlam[dlam != 0].min()
        S = int(0.4 / min_dl)
        fig = pl.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        for k, spine in ax.spines.items():
            spine.set_zorder(12.2)

        xs, ndx, dx = [0], 0, 0.001
        colors = ['r', 'g', '#7F38EC', '#9F000F', 'b', 'y']
        min_y, max_y = 0, 0

        lines = tuple()
        ## lv_names2 = [r'$Coulomb$', r'$vdWaals$'] ## for the paper
        lv_names2 = []
        for j in range(n_components):
            y = ave_dhdl[:, j]
            if not (y == 0).all():
                lv_names2.append(r'$%s$' % P.lv_names[j].capitalize())

        for j in range(n_components):

            y = ave_dhdl[:, j]
            if not (y == 0).all():

                # Get the coordinates.
                lj = lchange[:, j]
                x = lv[:, j][lj]
                y = y[lj] / P.beta_report

                if 'TI' in P.methods:
                    # Plot the TI integration area.
                    ss = 'TI'
                    for i in range(len(x) - 1):
                        min_y = min(y.min(), min_y)
                        max_y = max(y.max(), max_y)
                        # pl.plot(x,y)
                        if i % 2 == 0:
                            pl.fill_between(
                                x[i : i + 2] + ndx,
                                0,
                                y[i : i + 2],
                                color=colors[ndx],
                                alpha=1.0,
                            )
                        else:
                            pl.fill_between(
                                x[i : i + 2] + ndx,
                                0,
                                y[i : i + 2],
                                color=colors[ndx],
                                alpha=0.5,
                            )
                    xlegend = [-100 * wnum for wnum in range(len(lv_names2))]
                    pl.plot(
                        xlegend,
                        [0 * wnum for wnum in xlegend],
                        ls='-',
                        color=colors[ndx],
                        label=lv_names2[ndx],
                    )  ## for the paper

                    if 'TI-CUBIC' in P.methods and not cubspl[j] == 0:
                        # Plot the TI-CUBIC interpolation curve.
                        ss += ' and TI-CUBIC'
                        xnew = numpy.arange(0, 1 + dx, dx)
                        ynew = cubspl[j].interpolate(y, xnew)
                        min_y = min(ynew.min(), min_y)
                        max_y = max(ynew.max(), max_y)
                        pl.plot(
                            xnew + ndx,
                            ynew,
                            color='#B6B6B4',
                            ls='-',
                            solid_capstyle='round',
                            lw=3.0,
                        )

                else:
                    # Plot the TI-CUBIC integration area.
                    ss = 'TI-CUBIC'
                    for i in range(len(x) - 1):
                        xnew = numpy.arange(x[i], x[i + 1] + dx, dx)
                        ynew = cubspl[j].interpolate(y, xnew)
                        ynew[0], ynew[-1] = y[i], y[i + 1]
                        min_y = min(ynew.min(), min_y)
                        max_y = max(ynew.max(), max_y)
                        if i % 2 == 0:
                            pl.fill_between(
                                xnew + ndx, 0, ynew, color=colors[ndx], alpha=1.0
                            )
                        else:
                            pl.fill_between(
                                xnew + ndx, 0, ynew, color=colors[ndx], alpha=0.5
                            )

                # Store the abscissa values and update the subplot index.
                xs += (x + ndx).tolist()[1:]
                ndx += 1

        # Make sure the tick labels are not overcrowded.
        xs = numpy.array(xs)
        dl_mat = numpy.array([xs - i for i in xs])
        ri = range(len(xs))

        def getInd(r=ri, z=[0]):
            primo = r[0]
            min_dl = ndx * 0.02 * 2 ** (primo > 10)
            if dl_mat[primo].max() < min_dl:
                return z
            for i in r:
                for j in range(len(xs)):
                    if dl_mat[i, j] > min_dl:
                        z.append(j)
                        return getInd(ri[j:], z)

        print("K ====", K)
        xt = [i if (i in getInd()) else '' for i in range(K)]
        pl.xticks(xs[1:], xt[1:], fontsize=10)
        pl.yticks(fontsize=10)
        # ax = pl.gca()
        # for label in ax.get_xticklabels():
        #   label.set_bbox(dict(fc='w', ec='None', alpha=0.5))

        # Remove the abscissa ticks and set up the axes limits.
        for tick in ax.get_xticklines():
            tick.set_visible(False)
        pl.xlim(0, ndx)
        min_y *= 1.01
        max_y *= 1.01
        pl.ylim(min_y, max_y)

        for i, j in zip(xs[1:], xt[1:]):
            pl.annotate(
                ('%.2f' % (i - 1.0 if i > 1.0 else i) if not j == '' else ''),
                xy=(i, 0),
                xytext=(i, 0.01),
                size=10,
                rotation=90,
                textcoords=('data', 'axes fraction'),
                va='bottom',
                ha='center',
                color='#151B54',
            )
        if ndx > 1:
            lenticks = len(ax.get_ymajorticklabels()) - 1
            if min_y < 0:
                lenticks -= 1
            if lenticks < 5:
                from matplotlib.ticker import AutoMinorLocator as AML

                ax.yaxis.set_minor_locator(AML())
        pl.grid(which='both', color='w', lw=0.25, axis='y', zorder=12)
        pl.ylabel(
            r'$\mathrm{\langle{\frac{ \partial U } { \partial \lambda }}\rangle_{\lambda}\/%s}$'
            % P.units,
            fontsize=20,
            color='#151B54',
        )
        pl.annotate(
            '$\mathit{\lambda}$',
            xy=(0, 0),
            xytext=(0.5, -0.05),
            size=18,
            textcoords='axes fraction',
            va='top',
            ha='center',
            color='#151B54',
        )
        if not P.software.title() == 'Sire':
            lege = ax.legend(prop=FP(size=14), frameon=False, loc=1)
            for l in lege.legend_handles:
                l.set_linewidth(10)
        pl.savefig(os.path.join(P.output_directory, 'dhdl_TI.pdf'))
        pl.close(fig)
        return

    print("Plotting the free energy breakdown figure...")
    plotdFvsLambda1()
    plotdFvsLambda2()
    if 'TI' in P.methods or 'TI-CUBIC' in P.methods:
        print("Plotting the TI figure...")
        plotTI()


# ===================================================================================================
# FUNCTIONS: The Curve-Fitting Method. Called by the -c flag.
# ===================================================================================================


def plotCFM(u_kln, N_k, num_bins=100):
    """A graphical representation of what Bennett calls 'Curve-Fitting Method'."""

    print("Plotting the CFM figure...")

    def leaveTicksOnlyOnThe(xdir, ydir, axis):
        dirs = ['left', 'right', 'top', 'bottom']
        axis.xaxis.set_ticks_position(xdir)
        axis.yaxis.set_ticks_position(ydir)
        return

    def plotdg_vs_dU(yy, df_allk, ddf_allk):
        sq = (len(yy)) ** 0.5
        h = int(sq)
        w = h + 1 + 1 * (sq - h > 0.5)
        scale = round(w / 3.0, 1) + 0.4 if len(yy) > 13 else 1
        sf = numpy.ceil(scale * 3) if scale > 1 else 0
        fig = pl.figure(figsize=(8 * scale, 6 * scale))
        matplotlib.rc('axes', facecolor='#E3E4FA')
        matplotlib.rc('axes', edgecolor='white')
        if P.bSkipLambdaIndex:
            ks = [int(l) for l in P.bSkipLambdaIndex.split('-')]
            ks = numpy.delete(numpy.arange(K + len(ks)), ks)
        else:
            ks = range(K)
        for i, (xx_i, yy_i) in enumerate(yy):
            ax = pl.subplot(h, w, i + 1)
            ax.plot(xx_i, yy_i, color='r', ls='-', lw=3, marker='o', mec='r')
            leaveTicksOnlyOnThe('bottom', 'left', ax)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=6)
            ax.fill_between(
                xx_i,
                df_allk[i]['BAR'] - ddf_allk[i]['BAR'],
                df_allk[i]['BAR'] + ddf_allk[i]['BAR'],
                color='#D2B9D3',
                zorder=-1,
            )

            ax.annotate(
                r'$\mathrm{%d-%d}$' % (ks[i], ks[i + 1]),
                xy=(0.5, 0.9),
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(0, -2),
                size=14,
                textcoords='offset points',
                va='top',
                ha='center',
                color='#151B54',
                bbox=dict(fc='w', ec='none', boxstyle='round', alpha=0.5),
            )
            pl.xlim(xx_i.min(), xx_i.max())
        pl.annotate(
            r'$\mathrm{\Delta U_{i,i+1}\/(reduced\/units)}$',
            xy=(0.5, 0.03),
            xytext=(0.5, 0),
            xycoords=('figure fraction', 'figure fraction'),
            size=20 + sf,
            textcoords='offset points',
            va='center',
            ha='center',
            color='#151B54',
        )
        pl.annotate(
            r'$\mathrm{\Delta g_{i+1,i}\/(reduced\/units)}$',
            xy=(0.06, 0.5),
            xytext=(0, 0.5),
            rotation=90,
            xycoords=('figure fraction', 'figure fraction'),
            size=20 + sf,
            textcoords='offset points',
            va='center',
            ha='center',
            color='#151B54',
        )
        pl.savefig(os.path.join(P.output_directory, 'cfm.pdf'))
        pl.close(fig)
        return

    def findOptimalMinMax(ar):
        c = zip(*numpy.histogram(ar, bins=10))
        thr = int(ar.size / 8.0)
        mi, ma = ar.min(), ar.max()
        for i, j in c:
            if i > thr:
                mi = j
                break
        for i, j in c[::-1]:
            if i > thr:
                ma = j
                break
        return mi, ma

    def stripZeros(a, aa, b, bb):
        z = numpy.array([a, aa[:-1], b, bb[:-1]])
        til = 0
        for i, j in enumerate(a):
            if j > 0:
                til = i
                break
        z = z[:, til:]
        til = 0
        for i, j in enumerate(b[::-1]):
            if j > 0:
                til = i
                break
        z = z[:, : len(a) + 1 - til]
        a, aa, b, bb = z
        return a, numpy.append(aa, 100), b, numpy.append(bb, 100)

    K = len(u_kln)
    yy = []
    for k in range(0, K - 1):
        upto = min(N_k[k], N_k[k + 1])
        righ = -u_kln[k, k + 1, :upto]
        left = u_kln[k + 1, k, :upto]
        min1, max1 = findOptimalMinMax(righ)
        min2, max2 = findOptimalMinMax(left)

        mi = min(min1, min2)
        ma = max(max1, max2)

        (counts_l, xbins_l) = numpy.histogram(left, bins=num_bins, range=(mi, ma))
        (counts_r, xbins_r) = numpy.histogram(righ, bins=num_bins, range=(mi, ma))

        counts_l, xbins_l, counts_r, xbins_r = stripZeros(
            counts_l, xbins_l, counts_r, xbins_r
        )
        counts_r, xbins_r, counts_l, xbins_l = stripZeros(
            counts_r, xbins_r, counts_l, xbins_l
        )

        with numpy.errstate(divide='ignore', invalid='ignore'):
            log_left = numpy.log(counts_l) - 0.5 * xbins_l[:-1]
            log_righ = numpy.log(counts_r) + 0.5 * xbins_r[:-1]
            diff = log_left - log_righ
        yy.append((xbins_l[:-1], diff))

    plotdg_vs_dU(yy, df_allk, ddf_allk)
    return


# ===================================================================================================
# MAIN
# ===================================================================================================


def main():

    global dhdlt
    global u_klt
    global P
    global K
    global n_components
    global pymbar
    global dhdl
    global N_k
    global lv
    global dlam
    global ave_dhdl
    global std_dhdl
    global lchange
    global cubspl
    global mapl
    global u_kln
    global Deltaf_ij
    global dDeltaf_ij
    global df_allk
    global ddf_allk
    global nsnapshots
    global pl
    global FP
    global matplotlib

    # Timing.
    stime = ttt_time.time()
    print("Started on", ttt_time.asctime())
    print('Command line was:,', ' '.join(sys.argv))

    # Simulation profile P (to be stored in 'results.pickle') will amass information about the simulation.
    P = parser.parse_args()[0]

    P.methods = getMethods(P.methods.upper())
    P.units, P.beta, P.beta_report = checkUnitsAndMore(P.units)

    if (numpy.array([P.bForwrev, P.breakdown, P.bCFM, P.overlap]) != 0).any():
        import matplotlib  # 'matplotlib-1.1.0-1'; errors may pop up when using an earlier version. Add a version check?

        matplotlib.use('Agg')
        import matplotlib.pyplot as pl
        from matplotlib.font_manager import FontProperties as FP

    print(f"software: {P.software.title()}")
    if P.software.title() == 'Gromacs':
        import parser_gromacs

        nsnapshots, lv, dhdlt, u_klt = parser_gromacs.readDataGromacs(P)
        print('u_klt', u_klt, u_klt.shape)
    elif P.software.title() == 'Sire':
        import parser_sire

        nsnapshots, lv, dhdlt, u_klt = parser_sire.readDataSire(P)
    elif P.software.title() == 'Amber':
        import parser_amber

        nsnapshots, lv, dhdlt, u_klt = parser_amber.readDataAmber(P)
    elif P.software.title() == 'Desmond':
        import parser_desmond

        # NML: Desmond FEP jobs will always output with these names
        P.prefix = 'gibbs'
        P.suffix = 'dE'
        P.methods = ['BAR']  # Given only dE of adj Lambdas, limiting to BAR only
        nsnapshots, lv, u_klt = parser_desmond.readDataDesmond(P)
    else:
        from inspect import currentframe, getframeinfo

        lineno = getframeinfo(currentframe()).lineno
        print(
            "\n\n%s\n Looks like there is no yet proper parser to process your files. \n Please modify lines %d and %d of this script.\n%s\n\n"
        ) % (78 * "*", lineno + 3, lineno + 4, 78 * "*")
        #### LINES TO BE MODIFIED
        # import YOUR_OWN_FILE_PARSER
        # nsnapshots, lv, dhdlt, u_klt = YOUR_OWN_FILE_PARSER.yourDataParser(*args, **kwargs)
        #### All the four are numpy arrays.
        #### lv           is the array of lambda vectors.
        #### nsnapshots   is the number of equilibrated snapshots per each state.
        #### dhdlt[k,n,t] is the derivative of energy component n with respect to state k of snapshot t
        #### u_klt[k,m,t] is the reduced potential energy of snapshot t of state k evaluated at state m

    K, n_components = lv.shape
    lchange = get_lchange(lv)

    # NML: Check for all zeros in data files
    # sliu: change the all zero check to let the calculation continue if there is only dhdlt.
    all_zeros = not numpy.any(dhdlt) and not numpy.any(u_klt)

    if all_zeros == True:
        print("WARNING: Found all 0 in input data.")
        zero_output(K, P)
        if P.bForwrev:
            zero_dFt(K, P, nsnapshots)
        sys.exit("Exiting...")

    if (numpy.array(['Sire', 'Gromacs', 'Amber']) == P.software.title()).any():
        print(f"🔍 P.uncorr = {P.uncorr}")
        print(f"🔍 P.uncorr_threshold = {P.uncorr_threshold}")
        print(
            f"🔍 About to call uncorrelate with sta={numpy.zeros(K, int)}, fin={nsnapshots}"
        )

        dhdl, N_k, u_kln = uncorrelate(
            sta=numpy.zeros(K, int), fin=nsnapshots, do_dhdl=True
        )
    elif P.software.title() == 'Desmond':
        N_k, u_kln = uncorrelate(sta=numpy.zeros(K, int), fin=nsnapshots, do_dhdl=False)
    # Estimate free energy difference with MBAR -- all states at once.
    if 'MBAR' in P.methods:
        Deltaf_ij, dDeltaf_ij = estimatewithMBAR(
            u_kln, N_k, P.relative_tolerance, regular_estimate=True
        )

    # The TI preliminaries.
    if 'TI' in P.methods or 'TI-CUBIC' in P.methods:
        dlam, ave_dhdl, std_dhdl = TIprelim(lv)
    if 'TI-CUBIC' in P.methods:
        cubspl, mapl = getSplines(lchange)

    # Call other methods. Print stats. Store results.
    df_allk, ddf_allk = estimatePairs()

    totalEnergies()

    # Plot figures.
    if P.bForwrev:
        dF_t()
    if P.breakdown:
        plotdFvsLambda()
    if P.bCFM:
        if not (u_kln is None):
            plotCFM(u_kln, N_k, 50)

    print(
        "\nTime spent: %s hours, %s minutes, and %s seconds.\nFinished on %s",
        timeStatistics(stime),
    )


if __name__ == "__main__":
    #  import parser_gromacs
    #  import pickle
    main()
    #  P = parser.parse_args()[0]
    #  P.units, P.beta, P.beta_report = checkUnitsAndMore(P.units)
    #  nsnapshots, lv, dhdlt, u_klt = parser_gromacs.readDataGromacs(P)
    #  print("nsnapshots:", nsnapshots)
    #  print("lv:", lv)
    #  print("dhdlt:", dhdlt)
    #  print("u_klt:", u_klt)

    #  out = {
    #     "nsnapshots": nsnapshots,
    #     "lv"         : lv,
    #     "dhdlt"      : dhdlt,
    #     "u_klt"      : u_klt
    #  }

    #  with open("lambda_data.pkl", "wb") as f:
    #     pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print("Saved to fep_data.pkl")

# ===================================================================================================
#                                   End of the script
# ===================================================================================================
