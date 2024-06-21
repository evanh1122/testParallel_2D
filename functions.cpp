#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <unistd.h>
#include <armadillo>

// the following diagram shows how the grid will be split
//  ___________
// |_0_|_1_|_2_|
// |_3_|_4_|_5_|
// |_6_|_7_|_8_|

/// @brief generates a grid filled with random values. Automatically splits up the grid to each processor according
///        to the diagram above
/// @param width the bounds of the x axis of the grid
/// @param height the bounds of the y axis of the grid
/// @param interval the interval of the x and y values
/// @param iProc the rank of the processor that is running the function
/// @param nProcs the total number of processors
/// @return returns the grid that is generated
arma::mat genGrid(std::pair<double, double> width, std::pair<double, double> height, double interval, int iProc, int nProcs) {

    // calculates the width and height of a matrix that each processor is responsible for
    // adjusts for if width and/or height isn't perfectly divisible by the number of processors
    int root_nProcs = sqrt(nProcs);
    int subWidth = (abs(width.second - width.first + 1) / interval) / root_nProcs;
    if (subWidth % root_nProcs != 0) ++subWidth;

    int subHeight = (abs(height.second - height.first + 1) / interval) / root_nProcs;
    if (subHeight % root_nProcs != 0) ++subHeight;


    // if processor is responsible for an edge grid, then make sure that the grid doesn't go too far
    if (iProc % root_nProcs == root_nProcs - 1) subWidth = ((width.second + 1) - (subWidth * (root_nProcs - 1))) / interval;
    if (iProc / root_nProcs == root_nProcs - 1) subHeight = ((height.second + 1) - (subHeight * (root_nProcs - 1))) / interval;

    arma::mat grid(subHeight, subWidth, arma::fill::randu);

    return grid;
}



// memptr() points to the raw memory address
// or consider converting the matrix into something else like a vector of structs
// or take a look at the custom_send_recv function in Aether
//arma::mat findOwnership(arma::mat *grid, int width, int height) {
//    arma::mat grid(width, height, arma::fill::value(-1));
//    arma::mat globalGrid(width, height, arma::fill::value(-1));
//}


//arma::mat findCoeff() {
//}