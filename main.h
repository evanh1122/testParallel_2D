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
#include <fstream>

class SpatialGrid {
public:
    arma::mat grid;
    double ds; // Spatial resolution
    int row_start, col_start;
    int num_rows, num_cols;

    SpatialGrid(int rows, int cols, double resolution, int row_start = 0, int col_start = 0) 
        : grid(rows, cols), ds(resolution), row_start(row_start), col_start(col_start), num_rows(rows), num_cols(cols) {}

    void set(int i, int j, double temperature) {
        if (i >= 0 && i < grid.n_rows && j >= 0 && j < grid.n_cols) {
            grid(i, j) = temperature;
        } else {
            std::cout << "Invalid index in set: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid index in set");
        }
    }

    double get(int i, int j) {
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

    static int get_owner_proc_index(int i, int j, int total_side_length, int sqrt_procs, double resolution) {
        int num_points_per_side = total_side_length / resolution;
        int points_per_proc_side = num_points_per_side / sqrt_procs;
        int proc_row = i / points_per_proc_side;
        int proc_col = j / points_per_proc_side;
        return proc_col * sqrt_procs + proc_row;
    }

    static int get_owner_proc_coord(double x, double y, int total_side_length, int sqrt_procs, double resolution) {
        int num_points_per_side = total_side_length / resolution;
        int points_per_proc_side = num_points_per_side / sqrt_procs;
        int i = x / resolution;
        int j = y / resolution;
        int proc_row = i / points_per_proc_side;
        int proc_col = j / points_per_proc_side;
        return proc_col * sqrt_procs + proc_row;
    }

    std::vector<std::pair<double, double>> find_nearest_coords(double x, double y) {
        double i_unrounded = x / ds;
        double j_unrounded = y / ds;

        double x_below = -1, x_above = -1;
        double y_below = -1, y_above = -1;

        for (int i = 0; i < num_rows; i++) {
            double current_x = (row_start + i) * ds;
            if (current_x <= x) {
                x_below = current_x;
            }
            if (current_x >= x && x_above == -1) {
                x_above = current_x;
                break;
            }
        }

        for (int j = 0; j < num_cols; j++) {
            double current_y = (col_start + j) * ds;
            if (current_y <= y) {
                y_below = current_y;
            }
            if (current_y >= y && y_above == -1) {
                y_above = current_y;
                break;
            }
        }

        std::vector<std::pair<double, double>> nearest_coords;
        if (x_below != -1 && y_below != -1) {
            nearest_coords.push_back(std::make_pair(x_below, y_below));
        }
        if (x_below != -1 && y_above != -1) {
            nearest_coords.push_back(std::make_pair(x_below, y_above));
        }
        if (x_above != -1 && y_below != -1) {
            nearest_coords.push_back(std::make_pair(x_above, y_below));
        }
        if (x_above != -1 && y_above != -1) {
            nearest_coords.push_back(std::make_pair(x_above, y_above));
        }

        return nearest_coords;
    }

    void print() {
        for (int i = 0; i < grid.n_rows; i++) {
            for (int j = 0; j < grid.n_cols; j++) {
                double data = grid(i, j);
                std::cout << std::setw(10) << data << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_to_csv(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }

        for (int i = 0; i < grid.n_rows; i++) {
            for (int j = 0; j < grid.n_cols; j++) {
                double data = grid(i, j);
                file << data;
                if (j < grid.n_cols - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }

        file.close();
    }
};

#endif
