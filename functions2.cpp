// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 07/01/2024
// functions.cpp helper function file for testing two-dimensional examples of MPI programs for Aether

#include "main2.h"

#include <armadillo>

void printGlobalData(const SpatialGrid& local_grid, int nProcs, int iProc) {
    // Step 1: Prepare local data for sending
    // Convert local grid data to a format suitable for MPI communication
    // For simplicity, we're assuming all processors have the same amount of data to send
    int local_data_size = local_grid.num_rows * local_grid.num_cols;
    std::vector<double> local_temperatures(local_data_size);
    int index = 0;
    for (int i = 0; i < local_grid.num_rows; ++i) {
        for (int j = 0; j < local_grid.num_cols; ++j) {
            DataPoint data = local_grid.grid(i, j);
            // Replace -1 with actual temperature value
            local_temperatures[index++] = (data.temperature != -1) ? data.temperature : 0; // Assuming 0 is a placeholder for invalid data
        }
    }

    // Step 2: Gather data on root processor
    std::vector<double> global_temperatures;
    if (iProc == 0) {
        global_temperatures.resize(local_data_size * nProcs);
    }
    MPI_Gather(local_temperatures.data(), local_data_size, MPI_DOUBLE, global_temperatures.data(), local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 3: Print global data on root processor
    if (iProc == 0) {
        // Assuming global grid dimensions are known (global_num_rows, global_num_cols)
        int global_num_rows = std::sqrt(nProcs) * local_grid.num_rows;
        int global_num_cols = std::sqrt(nProcs) * local_grid.num_cols;
        for (int i = 0; i < global_num_rows; ++i) {
            for (int j = 0; j < global_num_cols; ++j) {
                std::cout << global_temperatures[i * global_num_cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}