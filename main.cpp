// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/22/2024
// main.cpp file for testing two-dimensional examples of MPI programs for Aether

#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++11 main.cpp functions.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug boolean to print testing statements **********
    bool debug_general = false;
    bool debug_print_transfer = true;
    // ***************************************************************

    MPI_Init(&argc, &argv); // Initialize MPI

    int nProcs, iProc; // Number of processors, processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get processor rank

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors



    MPI_Finalize(); // Finalize MPI
}