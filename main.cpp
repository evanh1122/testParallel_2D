// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/24/2024
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

    int iProc, nProcs; // Processor rank, number of processors
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc); // Get processor rank
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); // Get number of processors

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processors

    int sqrt = std::sqrt(nProcs);
    if (iProc == 0) {
        if (sqrt * sqrt != nProcs) {
            MPI_Finalize();
            throw std::runtime_error("Number of processors must be a perfect square");
        }
    }

    int total_side_length = 10;
    int side_length_per_proc = total_side_length / sqrt;

    int row_start = (iProc / sqrt) * side_length_per_proc;
    int col_start = (iProc % sqrt) * side_length_per_proc;

    double resolution = 1.0;

    SpatialGrid local_grid(side_length_per_proc, side_length_per_proc, resolution, row_start, col_start);

    
    for (int i = 0; i < side_length_per_proc; i++) {
        for (int j = 0; j < side_length_per_proc; j++) {
            auto [x_dist, y_dist] = local_grid.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = (2.0 * total_side_length) + total_side_length * std::sin(dist * 2.0 * M_PI / total_side_length);
            local_grid.set(i, j, DataPoint(temp));
        }
    }

    std::vector<DataPoint> local_data;
    for (int i = 0; i < side_length_per_proc; i++) {
        for (int j = 0; j < side_length_per_proc; j++) {
            local_data.push_back(local_grid.get(i, j));
        }
    }

    std::vector<DataPoint> global_data(total_side_length * total_side_length);

    MPI_Gather(local_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, global_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_grid(total_side_length, total_side_length, resolution);

        for (int proc = 0; proc < nProcs; proc++) {
            int row_start = (proc / sqrt) * side_length_per_proc;
            int col_start = (proc % sqrt) * side_length_per_proc;

            for (int i = 0; i < side_length_per_proc; i++) {
                for (int j = 0; j < side_length_per_proc; j++) {
                    global_grid.set(row_start + i, col_start + j, global_data[proc * side_length_per_proc * side_length_per_proc + i * side_length_per_proc + j]);
                }
            }
        }

        for (int i = 0; i < total_side_length; i++) {
            for (int j = 0; j < total_side_length; j++) {
                DataPoint data = global_grid.get(i, j);
                std::cout << std::setw(10) << data.temperature << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Finalize(); // Finalize MPI
}