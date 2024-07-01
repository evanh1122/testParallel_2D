#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp functions.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug boolean to print testing statements **********
    bool debug_general = true;
    // ***************************************************************

    MPI_Init(&argc, &argv); // Initialize MPI

    int iProc, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    MPI_Barrier(MPI_COMM_WORLD);

    int sqrt_procs = std::sqrt(nProcs);
    if (iProc == 0) {
        if (sqrt_procs * sqrt_procs != nProcs) {
            MPI_Finalize();
            throw std::runtime_error("Number of processors must be a perfect square");
        }
    }

    int total_side_length = 10;
    int side_length_per_proc = total_side_length / sqrt_procs;

    int row_start = (iProc / sqrt_procs) * side_length_per_proc;
    int col_start = (iProc % sqrt_procs) * side_length_per_proc;

    double resolution = 1;

    SpatialGrid local_grid(side_length_per_proc / resolution, side_length_per_proc / resolution, resolution, row_start, col_start);

    for (int i = 0; i < local_grid.num_rows; i++) {
        for (int j = 0; j < local_grid.num_cols; j++) {
            auto [x_dist, y_dist] = local_grid.get_global_coords(i, j);
            auto [x, y] = local_grid.get_local_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = (2.0 * total_side_length) + total_side_length * std::sin(dist * 2.0 * M_PI / total_side_length);
            local_grid.set(x, y, DataPoint(temp));
        }
    }

    std::vector<DataPoint> local_data;
    for (int i = 0; i < local_grid.num_rows; i++) {
        for (int j = 0; j < local_grid.num_cols; j++) {
            local_data.push_back(local_grid.get(i, j));
        }
    }

    std::vector<DataPoint> global_data(total_side_length * total_side_length / (resolution * resolution));

    MPI_Gather(local_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, global_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_grid(total_side_length / resolution, total_side_length / resolution, resolution);

        for (int proc = 0; proc < nProcs; proc++) {
            int row_start = (proc / sqrt_procs) * side_length_per_proc / resolution;
            int col_start = (proc % sqrt_procs) * side_length_per_proc / resolution;

            for (int i = 0; i < side_length_per_proc / resolution; i++) {
                for (int j = 0; j < side_length_per_proc / resolution; j++) {
                    global_grid.set((row_start + i) * global_grid.ds, (col_start + j) * global_grid.ds, global_data[proc * side_length_per_proc * side_length_per_proc / (resolution * resolution) + i * side_length_per_proc / resolution + j]);
                }
            }
        }

        global_grid.print(iProc);
    }

    MPI_Finalize(); // Finalize MPI
}
