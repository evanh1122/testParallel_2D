#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp functions.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug boolean to print testing statements **********
    bool debug_general = false;
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
    double resolution = 0.1;
    int num_points_per_side = total_side_length / resolution;
    int points_per_proc_side = num_points_per_side / sqrt_procs;

    int row_start = (iProc / sqrt_procs) * points_per_proc_side;
    int col_start = (iProc % sqrt_procs) * points_per_proc_side;

    SpatialGrid local_grid(points_per_proc_side, points_per_proc_side, resolution, row_start, col_start);

    for (int i = 0; i < local_grid.num_rows; i++) {
        for (int j = 0; j < local_grid.num_cols; j++) {
            auto [x_dist, y_dist] = local_grid.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_grid.set(i, j, DataPoint(temp));
        }
    }

    std::vector<DataPoint> local_data;
    for (int i = 0; i < local_grid.num_rows; i++) {
        for (int j = 0; j < local_grid.num_cols; j++) {
            local_data.push_back(local_grid.get(i, j));
        }
    }

    std::vector<DataPoint> global_data(num_points_per_side * num_points_per_side);

    MPI_Gather(local_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, global_data.data(), local_data.size() * sizeof(DataPoint), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_grid(num_points_per_side, num_points_per_side, resolution);

        for (int proc = 0; proc < nProcs; proc++) {
            int row_start = (proc / sqrt_procs) * points_per_proc_side;
            int col_start = (proc % sqrt_procs) * points_per_proc_side;

            for (int i = 0; i < points_per_proc_side; i++) {
                for (int j = 0; j < points_per_proc_side; j++) {
                    global_grid.set(row_start + i, col_start + j, global_data[proc * points_per_proc_side * points_per_proc_side + i * points_per_proc_side + j]);
                }
            }
        }

        if (debug_general) {
            global_grid.print();
        }
        global_grid.print_to_csv("output.csv");
    }

    MPI_Finalize(); // Finalize MPI
}
