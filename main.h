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
            std::cout << "Invalid index in get: (" << i << ", " << j << ")" << std::endl;
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
        double x_below = -1, x_above = -1;
        double y_below = -1, y_above = -1;

        for (int i = 0; i < num_rows; i++) {
            double current_x = (row_start + i) * ds;
            if (current_x <= x) {
                x_below = current_x;
            }
            if (current_x > x && x_above == -1) {
                x_above = current_x;
                break;
            }
        }

        for (int j = 0; j < num_cols; j++) {
            double current_y = (col_start + j) * ds;
            if (current_y <= y) {
                y_below = current_y;
            }
            if (current_y > y && y_above == -1) {
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

    static double bilinear_interpolation(SpatialGrid &grid, double x, double y) {
        auto coords = grid.find_nearest_coords(x, y);

        // Handle the case where coordinates align exactly with grid points
        if (coords.size() == 1) {
            return grid.get((coords[0].first - grid.row_start * grid.ds) / grid.ds, 
                            (coords[0].second - grid.col_start * grid.ds) / grid.ds);
        } else if (coords.size() < 4) {
            // Attempt to fill in missing coordinates if fewer than 4 points are found
            double x_below = -1, x_above = -1;
            double y_below = -1, y_above = -1;
            
            for (const auto &coord : coords) {
                if (coord.first < x && (x_below == -1 || coord.first > x_below)) {
                    x_below = coord.first;
                }
                if (coord.first > x && (x_above == -1 || coord.first < x_above)) {
                    x_above = coord.first;
                }
                if (coord.second < y && (y_below == -1 || coord.second > y_below)) {
                    y_below = coord.second;
                }
                if (coord.second > y && (y_above == -1 || coord.second < y_above)) {
                    y_above = coord.second;
                }
            }
            
            if (x_below == -1) x_below = x_above - grid.ds;
            if (x_above == -1) x_above = x_below + grid.ds;
            if (y_below == -1) y_below = y_above - grid.ds;
            if (y_above == -1) y_above = y_below + grid.ds;
            
            coords = {{x_below, y_below}, {x_below, y_above}, {x_above, y_below}, {x_above, y_above}};
        }

        double x1 = coords[0].first, y1 = coords[0].second;
        double x2 = coords[3].first, y2 = coords[3].second;

        // Handle edge case where x1 == x2 or y1 == y2
        if (x1 == x2) {
            x2 += grid.ds;
        }
        if (y1 == y2) {
            y2 += grid.ds;
        }

        int i1 = (x1 - grid.row_start * grid.ds) / grid.ds;
        int j1 = (y1 - grid.col_start * grid.ds) / grid.ds;
        int i2 = (x2 - grid.row_start * grid.ds) / grid.ds;
        int j2 = (y2 - grid.col_start * grid.ds) / grid.ds;

        // Boundary checks
        i1 = std::min(std::max(i1, 0), grid.num_rows - 1);
        j1 = std::min(std::max(j1, 0), grid.num_cols - 1);
        i2 = std::min(std::max(i2, 0), grid.num_rows - 1);
        j2 = std::min(std::max(j2, 0), grid.num_cols - 1);

        double Q11 = grid.get(i1, j1);
        double Q12 = grid.get(i1, j2);
        double Q21 = grid.get(i2, j1);
        double Q22 = grid.get(i2, j2);

        double R1 = ((x2 - x) / (x2 - x1)) * Q11 + ((x - x1) / (x2 - x1)) * Q21;
        double R2 = ((x2 - x) / (x2 - x1)) * Q12 + ((x - x1) / (x2 - x1)) * Q22;

        double P = ((y2 - y) / (y2 - y1)) * R1 + ((y - y1) / (y2 - y1)) * R2;

        return P;
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
