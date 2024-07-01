// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 07/01/2024
// main.h file for testing two-dimensional examples of MPI programs for Aether

#ifndef MAIN2_H
#define MAIN2_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <armadillo>

class DataPoint {
public:
    double temperature;
    DataPoint() : temperature(-1) {}
    DataPoint(double temp) : temperature(temp) {}
};

class SpatialGrid {
public:
    arma::field<DataPoint> grid;
    double ds;
    int num_side_divisions;
    int num_rows;
    int num_cols;
    int global_starting_index_row;
    int global_starting_index_col;
    int nProcs;
    int iProc;

    SpatialGrid(int global_num_rows, int global_num_cols, double resolution, int nProcs, int iProc) : grid(global_num_rows, global_num_cols), ds(resolution), nProcs(nProcs), iProc(iProc) {
        num_side_divisions = std::sqrt(nProcs);
        num_rows = global_num_rows / num_side_divisions;
        num_cols = global_num_cols / num_side_divisions;
        global_starting_index_row = (iProc / num_side_divisions) * num_rows;
        global_starting_index_col = (iProc % num_side_divisions) * num_cols;
    }

    void set(double global_x, double global_y, const DataPoint& data) {
        std::pair<int, int> indices = get_global_indices(global_x, global_y);
        int i = indices.first;
        int j = indices.second;
        if (owns_global_coord(global_x, global_y)) {
            grid(i, j) = data;
        } else {
            grid(i, j) = DataPoint(-1);
        }
    }

    DataPoint get(double global_x, double global_y) {
        std::pair<int, int> local_indices = get_local_indices(global_x, global_y);
        int local_i = local_indices.first;
        int local_j = local_indices.second;
        int owner;
        DataPoint data;

        if (owns_global_coord(global_x, global_y)) {
            data = grid(local_i, local_j);
            owner = iProc;
        }

        MPI_Bcast(&data, sizeof(DataPoint), MPI_BYTE, owner, MPI_COMM_WORLD);

        return data;
    }

    bool owns_global_coord(double global_x, double global_y) {
        int i = global_x / ds;
        int j = global_y / ds;
        int local_i = i - global_starting_index_row;
        int local_j = j - global_starting_index_col;
        return local_i >= 0 && local_i < num_rows && local_j >= 0 && local_j < num_cols;
    }

    std::pair<double, double> get_local_coords(int i, int j) {
        if (i >= 0 && i < num_rows && j >= 0 && j < num_cols) {
            return std::make_pair(i * ds, j * ds);
        } else {
            std::cout << "Invalid index in get_local_coords: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index in get_local_coords");
        }
    }

    std::pair<double, double> get_global_coords(int i, int j) {
        if (i >= 0 && i < num_rows && j >= 0 && j < num_cols) {
            return {(global_starting_index_row + i) * ds, (global_starting_index_col + j) * ds};
        } else {
            throw std::out_of_range("Grid indices out of bounds");
        }
    }

    std::pair<int, int> get_local_indices(double global_x, double global_y) {
        int i = global_x / ds;
        int j = global_y / ds;
        int local_i = i - global_starting_index_row;
        int local_j = j - global_starting_index_col;
        return std::make_pair(local_i, local_j);
    }

    std::pair<int, int> get_global_indices(double global_x, double global_y) {
        int i = global_x / ds;
        int j = global_y / ds;
        return std::make_pair(i, j);
    }

    void print(int print_proc = 0) {
        if (iProc == print_proc) {
            for (int i = 0; i < grid.n_rows; i++) {
                for (int j = 0; j < grid.n_cols; j++) {
                    DataPoint data = grid(i, j);
                    std::cout << std::setw(10) << data.temperature << " ";
                }
                std::cout << std::endl;
            }
        }
    }
};

#endif