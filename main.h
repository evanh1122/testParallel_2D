#ifndef MAIN_H
#define MAIN_H

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

    DataPoint() : temperature(0.0) {}
    DataPoint(double temp) : temperature(temp) {}
};

class SpatialGrid {
public:
    arma::field<DataPoint> grid;
    double ds; // Spatial resolution
    int row_start, col_start;
    int num_rows, num_cols;

    SpatialGrid(int rows, int cols, double resolution, int row_start = 0, int col_start = 0) 
        : grid(rows, cols), ds(resolution), row_start(row_start), col_start(col_start), num_rows(rows), num_cols(cols) {}

    void set(double x, double y, const DataPoint& data) {
        int i = x / ds;
        int j = y / ds;
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            grid(i, j) = data;
        } else {
            std::cout << "Invalid index in set: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index in set");
        }
    }

    DataPoint get(int i, int j) {
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            return grid(i, j);
        } else {
            throw std::out_of_range("Invalid index in get");
        }
    }

    std::pair<double, double> get_local_coords(int i, int j) {
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            return std::make_pair(i * ds, j * ds);
        } else {
            std::cout << "Invalid index in get_local_coords: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index in get_local_coords");
        }
    }

    std::pair<double, double> get_global_coords(int i, int j) {
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            return {(row_start + i) * ds, (col_start + j) * ds};
        } else {
            throw std::out_of_range("Grid indices out of bounds");
        }
    }

    void print(int iProc) {
        if (iProc == 0) {
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