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
        const double tolerance = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tolerance) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tolerance) / y_resolution + 0.5);
        if (i >= 0 && i < x_indices_per_proc && j >= 0 && j < y_indices_per_proc) {
            grid(i, j) = value;
        } else {
            std::cout << "Invalid coordinate in set: (" << x << ", " << y << ")" << std::endl;
            throw std::runtime_error("Invalid coordinate in set");
        }
    }

    double get(double x, double y) {
        const double tolerance = 1e-9;
        int i = static_cast<int>((x - x_start * x_resolution + tolerance) / x_resolution + 0.5);
        int j = static_cast<int>((y - y_start * y_resolution + tolerance) / y_resolution + 0.5);
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
        int x_start = static_cast<int>(x / x_res) / x_indices_per_proc;
        int y_start = static_cast<int>(y / y_res) / y_indices_per_proc;
        return y_start * sqrt_procs + x_start;
    }

    std::vector<std::pair<double, double>> find_nearest_coords(double x, double y, double src_x_resolution, double src_y_resolution, int src_total_width, int src_x_start, int src_y_start) {
        std::vector<std::pair<double, double>> nearest_coords;

        double x_below = -1, x_above = -1;
        double y_below = -1, y_above = -1;

        int num_x_indices = src_total_width / src_x_resolution;
        int num_y_indices = src_total_width / src_y_resolution;

        // Determine the global indices in the source grid
        int x_index = static_cast<int>(x / src_x_resolution);
        int y_index = static_cast<int>(y / src_y_resolution);

        // Find the nearest coordinates in the x direction
        if (x_index > 0) {
            x_below = (x_index - 1) * src_x_resolution;
        }
        if (x_index < num_x_indices - 1) {
            x_above = (x_index + 1) * src_x_resolution;
        }

        // Find the nearest coordinates in the y direction
        if (y_index > 0) {
            y_below = (y_index - 1) * src_y_resolution;
        }
        if (y_index < num_y_indices - 1) {
            y_above = (y_index + 1) * src_y_resolution;
        }

        // Add valid coordinates to the nearest_coords vector
        if (x_below != -1 && y_below != -1) nearest_coords.push_back({x_below, y_below});
        if (x_below != -1 && y_above != -1) nearest_coords.push_back({x_below, y_above});
        if (x_above != -1 && y_below != -1) nearest_coords.push_back({x_above, y_below});
        if (x_above != -1 && y_above != -1) nearest_coords.push_back({x_above, y_above});

        // Always include the exact point itself if it lies on a grid point
        nearest_coords.push_back({x_index * src_x_resolution, y_index * src_y_resolution});

        // Ensure we have exactly 4 points (even if some are repeated)
        if (nearest_coords.size() < 4) {
            if (x_below != -1) nearest_coords.push_back({x_below, y_index * src_y_resolution});
            if (x_above != -1) nearest_coords.push_back({x_above, y_index * src_y_resolution});
            if (y_below != -1) nearest_coords.push_back({x_index * src_x_resolution, y_below});
            if (y_above != -1) nearest_coords.push_back({x_index * src_x_resolution, y_above});
        }

        // If we still don't have enough points, add the point itself until we have 4 points
        while (nearest_coords.size() < 4) {
            nearest_coords.push_back({x_index * src_x_resolution, y_index * src_y_resolution});
        }

        return nearest_coords;
    }

    static void transfer_coord(int iProc, int nProcs, double x, double y, SpatialGrid &src_grid, SpatialGrid &dest_grid, bool print = false) {
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
            std::vector<std::pair<double, double> > nearest_coords = src_grid.find_nearest_coords(x, y, src_grid.x_resolution, src_grid.y_resolution, src_grid.total_spatial_width, src_grid.x_start, src_grid.y_start);

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
                if (dest_owner != iProc) {
                    MPI_Send(&top_left_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
                }
            }
            if (iProc == bottom_left_owner) {
                bottom_left_val = src_grid.get(x2, y2);
                if (dest_owner != iProc) {
                    MPI_Send(&bottom_left_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
                }
            }
            if (iProc == top_right_owner) {
                top_right_val = src_grid.get(x3, y3);
                if (dest_owner != iProc) {
                    MPI_Send(&top_right_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
                }
            }
            if (iProc == bottom_right_owner) {
                bottom_right_val = src_grid.get(x4, y4);
                if (dest_owner != iProc) {
                    MPI_Send(&bottom_right_val, 1, MPI_DOUBLE, dest_owner, 0, MPI_COMM_WORLD);
                }
            }
            if (iProc == dest_owner) {
                if (iProc != top_left_owner) {
                    MPI_Recv(&top_left_val, 1, MPI_DOUBLE, top_left_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (iProc != bottom_left_owner) {
                    MPI_Recv(&bottom_left_val, 1, MPI_DOUBLE, bottom_left_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (iProc != top_right_owner) {
                    MPI_Recv(&top_right_val, 1, MPI_DOUBLE, top_right_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                if (iProc != bottom_right_owner) {
                    MPI_Recv(&bottom_right_val, 1, MPI_DOUBLE, bottom_right_owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

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
        const double tolerance = 1e-9;  // Small tolerance to handle precision issues

        // Correctly validate against the entire grid, not just local
        if (x < 0 || x >= total_spatial_width || y < 0 || y >= total_spatial_width) {
            return false;
        }

        // Ensure the coordinates align with the grid resolution
        double x_mod = fmod(x, x_resolution);
        double y_mod = fmod(y, y_resolution);

        return (std::abs(x_mod) < tolerance || std::abs(x_mod - x_resolution) < tolerance) &&
            (std::abs(y_mod) < tolerance || std::abs(y_mod - y_resolution) < tolerance);
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