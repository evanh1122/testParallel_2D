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

// GEN_GRID_REF:
// Modifies grid_ref (armadillo matrix) to be a simple two-dimensional grid of points with a logically increasing format
// within a specified range. Each processor is assigned a chunk of grid_ref to own.
arma::mat gen_grid_ref(int iProc, int nProcs, std::pair<double, double> start_index, std::pair<double, double> end_index, int nRows, int nCols);

#endif