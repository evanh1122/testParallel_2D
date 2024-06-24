// Author: Benjamen Miller, University of Michigan - Ann Arbor
// Date: 06/24/2024
// main.h file for testing two-dimensional examples of MPI programs for Aether

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

    SpatialGrid(int rows, int cols, double resolution) : grid(rows, cols), ds(resolution) {}

    void set(double x, double y, const DataPoint& data) {
        int i = x / ds;
        int j = y / ds;
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            grid(i, j) = data;
        } else {
            std::cout << "Invalid index: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index");
        }
    }

    DataPoint get(double x, double y) {
        int i = x / ds;
        int j = y / ds;
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            return grid(i, j);
        } else {
            std::cout << "Invalid index: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index");
        }
    }

    std::pair<double, double> get_coords(int i, int j) {
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            return std::make_pair(i * ds, j * ds);
        } else {
            std::cout << "Invalid index: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index");
        }
    }
};

#endif