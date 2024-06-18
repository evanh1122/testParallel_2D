// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/18/2024
// main.cpp file for testing two-dimensional examples of MPI programs for Aether

#include "main2.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++11 main2.cpp functions2.cpp -o main2
// RUN COMMAND: mpirun -np 4 ./main2

int main(int argc, char **argv) {
    bool debug = true;

    MPI_Init(&argc, &argv); // Initialize MPI

    int nProcs, iProc; // Number of processors, processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get processor rank

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    int grid_ref_nRows = 12;
    int grid_ref_nCols = 12;
    std::pair<double, double> grid_ref_start_index = {0, 0};
    std::pair<double, double> grid_ref_end_index = {11, 11};

    arma::mat grid_ref = gen_grid_ref(iProc, nProcs, grid_ref_start_index, grid_ref_end_index, grid_ref_nRows, grid_ref_nCols);

    std::cout << grid_ref << std::endl;

    MPI_Finalize(); // Finalize MPI
}