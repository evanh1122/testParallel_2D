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
    double x_resolution, y_resolution;
    int x_start, y_start;
    int x_indices_per_proc, y_indices_per_proc;
    int total_spatial_width;

    SpatialGrid(int x_indices_per_proc, int y_indices_per_proc, int total_spatial_width, double x_resolution, double y_resolution, int x_start = 0, int y_start = 0) {
        this->grid = arma::zeros<arma::mat>(x_indices_per_proc, y_indices_per_proc);
        this->x_resolution = x_resolution;
        this->y_resolution = y_resolution;
        this->x_start = x_start;
        this->y_start = y_start;
        this->x_indices_per_proc = x_indices_per_proc;
        this->y_indices_per_proc = y_indices_per_proc;
        this->total_spatial_width = total_spatial_width;
    }

    void set(double x, double y, double value) {
        int i = static_cast<int>((x - x_start * x_resolution) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            grid(i, j) = value;
        } else {
            std::cout << "Invalid coordinate in set: (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Invalid coordinate in set");
        }
    }

    double get(double x, double y) {
        int i = static_cast<int>((x - x_start * x_resolution) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            return grid(i, j);
        } else {
            std::cout << "Invalid coordinate in get: (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Invalid coordinate in get");
        }
    }

    std::pair<double, double> get_global_coords(int i, int j) {
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            return {(x_start + i) * x_resolution, (y_start + j) * y_resolution};
        } else {
            std::cout << "Invalid indices in get_global_coords: (" << i << ", " << j << ")" << std::endl;
            throw std::runtime_error("Invalid indices in get_global_coords");
        }
    }

    int get_owner_coord(double x, double y, int sqrt_procs, double x_res, double y_res) {
        int num_x_indices = total_spatial_width / x_res;
        int num_y_indices = total_spatial_width / y_res;
        int x_indices_per_proc = num_x_indices / sqrt_procs;
        int y_indices_per_proc = num_y_indices / sqrt_procs;
        int x_start = (x / x_res) / x_indices_per_proc;
        int y_start = (y / y_res) / y_indices_per_proc;
        return y_start * sqrt_procs + x_start;
    }

    std::vector<std::pair<double, double> > find_nearest_coords(double x, double y) {
        std::vector<std::pair<double, double> > nearest_coords;
        double x_below = -1, x_above = -1;
        double y_below = -1, y_above = -1;

        for (int i = 0; i < x_indices_per_proc; i++) {
            double current_x = (x_start + i) * x_resolution;
            if (current_x <= x) {
                x_below = current_x;
            } else {
                x_above = current_x;
                break;
            }
        }

        for (int j = 0; j < y_indices_per_proc; j++) {
            double current_y = (y_start + j) * y_resolution;
            if (current_y <= y) {
                y_below = current_y;
            } else {
                y_above = current_y;
                break;
            }
        }

        if (x_below != -1 && y_below != -1) {
            nearest_coords.push_back({x_below, y_below});
        }
        if (x_below != -1 && y_above != -1) {
            nearest_coords.push_back({x_below, y_above});
        }
        if (x_above != -1 && y_below != -1) {
            nearest_coords.push_back({x_above, y_below});
        }
        if (x_above != -1 && y_above != -1) {
            nearest_coords.push_back({x_above, y_above});
        }

        return nearest_coords;
    }

    void transfer_coord(int iProc, int nProcs, double x, double y, SpatialGrid &src_grid, SpatialGrid &dest_grid, bool print = false) {
        bool src_exists = src_grid.is_valid_coord(x, y);
        bool dest_exists = dest_grid.is_valid_coord(x, y);

        if (src_exists && dest_exists) {
            int src_owner = src_grid.get_owner_coord(x, y, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int dest_owner = dest_grid.get_owner_coord(x, y, std::sqrt(nProcs), dest_grid.x_resolution, dest_grid.y_resolution);

            if (src_owner == iProc && dest_owner == iProc) {
                double value = src_grid.get(x, y);
                dest_grid.set(x, y, value);
            } else if (src_owner == iProc && dest_owner != iProc) {
                double value = src_grid.get(x, y);
                MPI_Send(&value, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
            } else if (src_owner != iProc && dest_owner == iProc) {
                double value;
                MPI_Recv(&value, 1, MPI_DOUBLE, src_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                dest_grid.set(x, y, value);
            } else {
                if (print) {
                    std::cout << "Processor " << iProc << " does not own the source or destination index. Skipping..." << std::endl;
                }
            }
        } else if (!src_exists && dest_exists) {
            std::vector<std::pair<double, double> > nearest_coords = src_grid.find_nearest_coords(x, y);

            if (nearest_coords.size() < 4) {
                std::cout << "Error: Not enough points for interpolation" << std::endl;
            }

            double x1 = nearest_coords[0].first, y1 = nearest_coords[0].second;
            double x2 = nearest_coords[1].first, y2 = nearest_coords[1].second;
            double x3 = nearest_coords[2].first, y3 = nearest_coords[2].second;
            double x4 = nearest_coords[3].first, y4 = nearest_coords[3].second;

            int top_left_owner = src_grid.get_owner_coord(x1, y1, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int bottom_left_owner = src_grid.get_owner_coord(x2, y2, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int top_right_owner = src_grid.get_owner_coord(x3, y3, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);
            int bottom_right_owner = src_grid.get_owner_coord(x4, y4, std::sqrt(nProcs), src_grid.x_resolution, src_grid.y_resolution);

            int dest_owner = dest_grid.get_owner_coord(x, y, std::sqrt(nProcs), dest_grid.x_resolution, dest_grid.y_resolution);

            double top_left_val, bottom_left_val, top_right_val, bottom_right_val;

            if (iProc == top_left_owner) {
                top_left_val = src_grid.get(x1, y1);
                MPI_Send(&top_left_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
            }
            if (iProc == bottom_left_owner) {
                bottom_left_val = src_grid.get(x2, y2);
                MPI_Send(&bottom_left_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
            }
            if (iProc == top_right_owner) {
                top_right_val = src_grid.get(x3, y3);
                MPI_Send(&top_right_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
            }
            if (iProc == bottom_right_owner) {
                bottom_right_val = src_grid.get(x4, y4);
                MPI_Send(&bottom_right_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
            }
            if (iProc == dest_owner) {
                MPI_Recv(&top_left_val, 1, MPI_DOUBLE, top_left_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&bottom_left_val, 1, MPI_DOUBLE, bottom_left_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&top_right_val, 1, MPI_DOUBLE, top_right_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&bottom_right_val, 1, MPI_DOUBLE, bottom_right_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                double t = (x - x1) / (x2 - x1);
                double u = (y - y1) / (y3 - y1);
                double interpolated_value = (1 - t) * (1 - u) * top_left_val + t * (1 - u) * top_right_val + (1 - t) * u * bottom_left_val + t * u * bottom_right_val;
                dest_grid.set(x, y, interpolated_value);
            }
        } else if (src_exists && !dest_exists) {
            std::cout << "Destination grid does not contain coordinate (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Destination grid does not contain coordinate");
        } else {
            std::cout << "Neither source nor destination grid contain coordinate (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Neither source nor destination grid contain coordinate");
        }
    }

    bool is_valid_coord(double x, double y) {
        if (x < x_start * x_resolution || x >= (x_start + x_indices_per_proc) * x_resolution || y < y_start * y_resolution || y >= (y_start + y_indices_per_proc) * y_resolution) {
            return false;
        }

        double x_mod = fmod(x - x_start * x_resolution, x_resolution);
        double y_mod = fmod(y - y_start * y_resolution, y_resolution);
        return x_mod == 0.0 && y_mod == 0.0;
    }

    void print_to_terminal() {
        for (int i = 0; i < x_indices_per_proc; i++) {
            for (int j = 0; j < y_indices_per_proc; j++) {
                double data = grid(i, j);
                std::cout << std::setw(10) << data << " ";
            }
            std::cout << std::endl;
        }
    }

    void print_to_csv(const std::string &filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }

        for (int i = 0; i < x_indices_per_proc; i++) {
            for (int j = 0; j < y_indices_per_proc; j++) {
                double data = grid(i, j);
                file << data;
                if (j < y_indices_per_proc - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }

        file.close();
    }
};

#endif