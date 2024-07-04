#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug booleans to print testing statements **********
    bool debug_global_grid = true; // Creates global grid and prints to output file
    bool print_global_grid = false; // Prints global grid to console
    bool test_nearest_coords = false; // Tests find_nearest_coords function
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
    double resolution = 0.5;
    int num_points_per_side = total_side_length / resolution;
    int points_per_proc_side = num_points_per_side / sqrt_procs;

    int row_start = (iProc / sqrt_procs) * points_per_proc_side;
    int col_start = (iProc % sqrt_procs) * points_per_proc_side;

    SpatialGrid local_grid_temp(points_per_proc_side, points_per_proc_side, resolution, row_start, col_start);

    for (int i = 0; i < local_grid_temp.num_rows; i++) {
        for (int j = 0; j < local_grid_temp.num_cols; j++) {
            auto [x_dist, y_dist] = local_grid_temp.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_grid_temp.set(i, j, temp);
        }
    }

    if (debug_global_grid) {
        std::vector<double> local_data_temp;
        for (int i = 0; i < local_grid_temp.num_rows; i++) {
            for (int j = 0; j < local_grid_temp.num_cols; j++) {
                local_data_temp.push_back(local_grid_temp.get(i, j));
            }
        }

        std::vector<double> global_data_temp(num_points_per_side * num_points_per_side);

        MPI_Gather(local_data_temp.data(), local_data_temp.size() * sizeof(double), MPI_BYTE, global_data_temp.data(), local_data_temp.size() * sizeof(double), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (iProc == 0) {
            SpatialGrid global_grid_temp(num_points_per_side, num_points_per_side, resolution);

            for (int proc = 0; proc < nProcs; proc++) {
                int row_start = (proc / sqrt_procs) * points_per_proc_side;
                int col_start = (proc % sqrt_procs) * points_per_proc_side;

                for (int i = 0; i < points_per_proc_side; i++) {
                    for (int j = 0; j < points_per_proc_side; j++) {
                        global_grid_temp.set(row_start + i, col_start + j, global_data_temp[proc * points_per_proc_side * points_per_proc_side + i * points_per_proc_side + j]);
                    }
                }
            }

            if (print_global_grid) {
                global_grid_temp.print();
            }
            global_grid_temp.print_to_csv("output.csv");

            if (test_nearest_coords) {
                std::vector<std::pair<double, double> > nearest_coords = global_grid_temp.find_nearest_coords(1.5245, 2.3254);
                std::cout << "Nearest coords to (1.5245, 2.3254):" << std::endl;
                for (auto [i, j] : nearest_coords) {
                    std::cout << "(" << i << ", " << j << ")" << std::endl;
                }
            }
        }
    }

    MPI_Finalize(); // Finalize MPI
}
