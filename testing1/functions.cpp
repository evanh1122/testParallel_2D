// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/22/2024
// functions.cpp helper function file for testing two-dimensional examples of MPI programs for Aether

#include "main.h"

#include <armadillo>

// GEN_TEMP_REF:
// Function to generate an Armadillo matrix filled with temperature values based on index in the matrix
arma::mat gen_temp_ref(int x_size, int y_size) {
    int iProc, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    int num_points = x_size * y_size;
    int n = std::sqrt(nProcs);
    if (n * n != nProcs) {
        // Handle the case when nProcs is not a perfect square
        // For example, you can print an error message and return an empty matrix
        std::cerr << "Error: Number of processes is not a perfect square." << std::endl;
        return arma::mat();
    }
    int num_rows = x_size / n;
    int num_cols = y_size / n;

    int start_row = (iProc / n) * num_rows;
    int start_col = (iProc % n) * num_cols;

    arma::mat temp_ref(x_size, y_size);
    temp_ref.fill(-1);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int global_row = start_row + i;
            int global_col = start_col + j;
            if (global_row < x_size && global_col < y_size) {
                int dist_x = global_row - 0;
                int dist_y = global_col - 0;
                double grid_ref = std::sqrt(dist_x * dist_x + dist_y * dist_y);
                double temp_val = 200.0 + 100.0 * std::sin(grid_ref * 2.0 * M_PI / num_points);
                temp_ref(global_row, global_col) = temp_val;
            }
        }
    }

    return temp_ref;
}

// GEN_GLOBAL_MAT:
// Produces a matrix of all temperature values across all processors using MPI_Allreduce
arma::mat gen_global_mat(arma::mat local_mat) {
    int iProc, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    arma::mat global_mat(local_mat.n_rows, local_mat.n_cols);
    global_mat.fill(-1);

    // Combine all processor's matrices into a global matrix of temperature values
    MPI_Allreduce(local_mat.memptr(), global_mat.memptr(), local_mat.n_rows * local_mat.n_cols, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_mat;
}

// FIND_NEAREST_INDICES:
// This function takes in a source matrix and an index.
// The function searches through the source matrix to find the four closest indices to the given index
// and returns them in a vector of format <x_low, x_high, y_low, y_high>. If the index exists on the source map
// then the function will return the index itself as the closest index in the format <x_index, y_index, -1, -1>.
