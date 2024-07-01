// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 07/01/2024
// main.cpp file for testing two-dimensional examples of MPI programs for Aether

#include "main2.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++11 main2.cpp functions2.cpp -o main2
// RUN COMMAND (4 Processors): mpirun -np 4 ./main2

int main(int argc, char **argv) {
    // ********** Debug boolean to print testing statements **********
    bool debug_general = true;
    // ***************************************************************
    int global_num_rows = 10;
    int global_num_cols = 10;
    double resolution = 1;

    MPI_Init(&argc, &argv); // Initialize MPI

    int iProc, nProcs; // Processor rank, number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get number of processors

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    SpatialGrid local_grid(global_num_rows, global_num_cols, resolution, nProcs, iProc);

    for (int i = 0; i < local_grid.num_rows; ++i) {
        for (int j = 0; j < local_grid.num_cols; ++j) {
            auto global_coords = local_grid.get_global_coords(i, j);
            double global_x = global_coords.first;
            double global_y = global_coords.second;

            double distance = sqrt(pow(global_x, 2) + pow(global_y, 2));

            double temp = (2.0 * global_num_rows) + global_num_rows * std::sin(distance * 2.0 * M_PI / global_num_rows);

            DataPoint data(temp);

            local_grid.set(global_x, global_y, data);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    local_grid.print();

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    if (iProc == 0) {
        std::cout << "******************" << std::endl;
    }

    std::vector<DataPoint> local_data;
    for (int i = 0; i < local_grid.num_rows; ++i) {
        for (int j = 0; j < local_grid.num_cols; ++j) {
            local_data.push_back(local_grid.get(i, j));
        }
    }

    std::vector<DataPoint> global_data;
    if (iProc == 0) {
        global_data.resize(global_num_rows * global_num_cols);
    }

    MPI_Gather(local_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE,
               global_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_grid(global_num_rows, global_num_cols, resolution, nProcs, iProc);

        for (int proc = 0; proc < nProcs; proc++) {
            int sqrt_nProcs = std::sqrt(nProcs);
            int row_start = (proc / sqrt_nProcs) * global_num_rows / sqrt_nProcs;
            int col_start = (proc % sqrt_nProcs) * global_num_cols / sqrt_nProcs;

            for (int i = 0; i < global_num_rows / sqrt_nProcs; ++i) {
                for (int j = 0; j < global_num_cols / sqrt_nProcs; ++j) {
                    int global_idx = proc * (global_num_rows * global_num_cols / nProcs) + i * (global_num_cols / sqrt_nProcs) + j;
                    global_grid.set((row_start + i) * resolution, (col_start + j) * resolution, global_data[global_idx]);
                }
            }
        }

        global_grid.print();
    }
    
    MPI_Finalize();
}