// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/18/2024
// functions.cpp helper function file for testing two-dimensional examples of MPI programs for Aether

#include "main2.h"

// GEN_GRID_REF:
// Modifies grid_ref (armadillo matrix) to be a simple two-dimensional grid of points with a logically increasing format
// within a specified range. Each processor is assigned a chunk of grid_ref to own.
arma::mat gen_grid_ref(int iProc, int nProcs, std::pair<double, double> start_index, std::pair<double, double> end_index, int nRows, int nCols) {
    // Calculate the number of rows and columns each processor will own
    int nRows_per_proc = nRows / nProcs;
    int nCols_per_proc = nCols / nProcs;

    // Calculate the starting and ending indices for the processor
    int start_row = iProc * nRows_per_proc + start_index.first;
    int end_row = start_row + nRows_per_proc - 1 + end_index.first;
    int start_col = iProc * nCols_per_proc + start_index.second;
    int end_col = start_col + nCols_per_proc - 1 + end_index.second;

    // Initialize the grid_ref matrix
    arma::mat grid_ref(nRows, nCols, arma::fill::ones);
    grid_ref *= -1;

    

    return grid_ref;
}