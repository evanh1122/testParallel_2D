// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/22/2024
// main.cpp file for testing two-dimensional examples of MPI programs for Aether

#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++11 main.cpp functions.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug boolean to print testing statements **********
    bool debug_general = true;
    // ***************************************************************

    MPI_Init(&argc, &argv); // Initialize MPI

    int nProcs, iProc; // Number of processors, processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get processor rank

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    // ********** Generate temperature reference matrix **********
    int x_size = 10, y_size = 10; // Size of the temperature matrix

    arma::mat temp_ref = gen_temp_ref(x_size, y_size); // Generate temperature reference matrix

    

    int which_proc_prints = 0;
    if (debug_general) {
        if (iProc == which_proc_prints) {
            std::cout << "Temperature Reference Matrix:" << std::endl;
            std::cout << temp_ref << std::endl;
        }
    }

    arma::mat global_mat = gen_global_mat(temp_ref); // Generate global temperature matrix
    if (debug_general) {
        if (iProc == which_proc_prints) {
            std::cout << "Global Temperature Matrix:" << std::endl;
            std::cout << global_mat << std::endl;
        }
    }

    MPI_Finalize(); // Finalize MPI
}