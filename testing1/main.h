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

// GEN_TEMP_REF:
// Function to generate an Armadillo matrix filled with temperature values based on index in the matrix
arma::mat gen_temp_ref(int x_size, int y_size);

// GEN_GLOBAL_MAT:
// Produces a matrix of all temperature values across all processors using MPI_Allreduce
arma::mat gen_global_mat(arma::mat local_mat);

#endif