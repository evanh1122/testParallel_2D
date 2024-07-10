#include "main.h"
#include "mpi.h"

// COMPILE COMMAND: mpic++ -std=c++17 main.cpp -o main
// RUN COMMAND (4 Processors): mpirun -np 4 ./main

int main(int argc, char **argv) {
    // ********** Debug booleans to print testing statements **********
    bool debug_global_grid = true; // Creates global grid and prints to output file
    bool print_global_grid = false; // Prints global grid to console
    bool test_nearest_coords = false; // Tests find_nearest_coords function w/ print statement
    bool test_bilinear_interpolation = false; // Tests bilinear_interpolation function w/ print statement
    bool print_global_grid_new = false; // Prints new global grid to console
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

    SpatialGrid local_grid_temp(points_per_proc_side + 2, points_per_proc_side + 2, resolution, row_start, col_start);

    for (int i = 1; i <= points_per_proc_side; i++) {
        for (int j = 1; j <= points_per_proc_side; j++) {
            auto [x_dist, y_dist] = local_grid_temp.get_global_coords(i, j);
            double dist = std::sqrt(x_dist * x_dist + y_dist * y_dist);
            double temp = 200 + 100 * std::sin(dist * 25.0 * M_PI / 100);
            local_grid_temp.set(i, j, temp);
        }
    }

    local_grid_temp.exchange_halo(MPI_COMM_WORLD, iProc, sqrt_procs, points_per_proc_side);

    if (debug_global_grid) {
        std::vector<double> local_data_temp;
        for (int i = 1; i <= points_per_proc_side; i++) {
            for (int j = 1; j <= points_per_proc_side; j++) {
                local_data_temp.push_back(local_grid_temp.get(i, j));
            }
        }

        std::vector<double> global_data_temp(num_points_per_side * num_points_per_side);

        MPI_Gather(local_data_temp.data(), local_data_temp.size(), MPI_DOUBLE, global_data_temp.data(), local_data_temp.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (iProc == 0) {
            SpatialGrid global_grid_temp(num_points_per_side + 2, num_points_per_side + 2, resolution);

            for (int proc = 0; proc < nProcs; proc++) {
                int row_start = (proc / sqrt_procs) * points_per_proc_side;
                int col_start = (proc % sqrt_procs) * points_per_proc_side;

                for (int i = 0; i < points_per_proc_side; i++) {
                    for (int j = 0; j < points_per_proc_side; j++) {
                        global_grid_temp.set(row_start + i + 1, col_start + j + 1, global_data_temp[proc * points_per_proc_side * points_per_proc_side + i * points_per_proc_side + j]);
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

            if (test_bilinear_interpolation) {
                double interpolated_temp = global_grid_temp.bilinear_interpolation(global_grid_temp, 1.5245, 2.3254);
                std::cout << "Interpolated temperature at (1.5245, 2.3254): " << interpolated_temp << std::endl;
            }
        }
    }

    // ********** We are going to try to create a new SpatialGrid and fill it with all interpolated values... **********
    int new_total_side_length = 10;
    double new_resolution = 0.01;
    int new_num_points_per_side = new_total_side_length / new_resolution;
    int new_points_per_proc_side = new_num_points_per_side / sqrt_procs;

    int new_row_start = (iProc / sqrt_procs) * new_points_per_proc_side;
    int new_col_start = (iProc % sqrt_procs) * new_points_per_proc_side;

    SpatialGrid local_grid_new(new_points_per_proc_side + 2, new_points_per_proc_side + 2, new_resolution, new_row_start, new_col_start);

    for (int i = 1; i <= new_points_per_proc_side; i++) {
        for (int j = 1; j <= new_points_per_proc_side; j++) {
            double x, y;
            std::tie(x, y) = local_grid_new.get_global_coords(i, j);
            double temp = local_grid_temp.bilinear_interpolation(local_grid_temp, x, y);
            local_grid_new.set(i, j, temp);
        }
    }

    local_grid_new.exchange_halo(MPI_COMM_WORLD, iProc, sqrt_procs, new_points_per_proc_side);

    std::vector<double> local_data_new;
    for (int i = 1; i <= new_points_per_proc_side; i++) {
        for (int j = 1; j <= new_points_per_proc_side; j++) {
            local_data_new.push_back(local_grid_new.get(i, j));
        }
    }

    std::vector<double> global_data_new(new_num_points_per_side * new_num_points_per_side);

    MPI_Gather(local_data_new.data(), local_data_new.size(), MPI_DOUBLE, global_data_new.data(), local_data_new.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (iProc == 0) {
        SpatialGrid global_grid_new(new_num_points_per_side + 2, new_num_points_per_side + 2, new_resolution);

        for (int proc = 0; proc < nProcs; proc++) {
            int row_start = (proc / sqrt_procs) * new_points_per_proc_side;
            int col_start = (proc % sqrt_procs) * new_points_per_proc_side;

            for (int i = 0; i < new_points_per_proc_side; i++) {
                for (int j = 0; j < new_points_per_proc_side; j++) {
                    global_grid_new.set(row_start + i + 1, col_start + j + 1, global_data_new[proc * new_points_per_proc_side * new_points_per_proc_side + i * new_points_per_proc_side + j]);
                }
            }
        }

        if (print_global_grid_new) {
            global_grid_new.print();
        }
        global_grid_new.print_to_csv("output_new.csv");
    }

    MPI_Finalize(); // Finalize MPI
}