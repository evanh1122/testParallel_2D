#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <unistd.h>
#include <armadillo>

class Grid {

private:
    arma::mat grid;
    std::map<double, double> x, y, ownership;
    std::pair<double, double> width, height;
    double interval;
    int nProcs;

public:

    // the following diagram shows how the grid will be split
    //  ___________
    // |_0_|_1_|_2_|
    // |_3_|_4_|_5_|
    // |_6_|_7_|_8_|

    /// @brief generates a grid filled with random values. Automatically splits up the grid to each processor 
    ///        according to the diagram above
    /// @param width the bounds of the x axis of the grid
    /// @param height the bounds of the y axis of the grid
    /// @param interval the interval of the x and y values
    /// @param iProc the rank of the processor that is running the function
    /// @param nProcs the total number of processors

    // TO DO: create some type of onwership array since we know how the indecis are split
    // TO DO: create the x and y maps that tell you what coordinate correlates to what index
    Grid(std::pair<double, double> _width, std::pair<double, double> _height, double _interval, int iProc, 
         int _nProcs) : width(_width), height(_height), interval(_interval), nProcs(_nProcs) {

        // calculates the width and height of a matrix that each processor is responsible for
        // adjusts for if width and/or height isn't perfectly divisible by the number of processors
        int root_nProcs = sqrt(nProcs);
        int subWidth = (abs(width.second - width.first + interval) / interval) / root_nProcs;
        if (fmod(abs(width.second - width.first + interval) / interval, root_nProcs) != 0) ++subWidth;

        int subHeight = (abs(height.second - height.first + interval) / interval) / root_nProcs;
        if (fmod(abs(height.second - height.first + interval) / interval, root_nProcs) != 0) ++subHeight;


        // if processor is responsible for an edge grid, then make sure that the grid doesn't go too far
        if (iProc % root_nProcs == root_nProcs - 1) 
            subWidth = ((width.second + interval) / interval) - (subWidth * (root_nProcs - 1));

        if (iProc / root_nProcs == root_nProcs - 1) 
            subHeight = ((height.second + interval) / interval) - (subHeight * (root_nProcs - 1));

        grid.set_size(subHeight, subWidth);
        grid.fill(arma::fill::randu);


        // fill in the x and y maps that correspond x and y positions to indices of the matrix
        // TO DO: this part currently doesn't work with intervals other than 1...
        //        Maybe consider doing the ownership array first, then we can use that to figure out 
        //        which processor is responsible for what parts of the grid, making this part easier
        int iMat = 0;
        for (int iMap = width.first + (iProc * subWidth); iMap < width.first + ((iProc + 1) * subWidth) && 
             iMap <= width.second; iMap += interval) {
                x[iMap] = iMat;
                ++iMat;
        }

        iMat = 0;
        for (int iMap = height.first + (iProc * subHeight); iMap < height.first + ((iProc + 1) * subHeight) && 
             iMap <= height.second; iMap += interval) {
                y[iMap] = iMat;
                ++iMat;
        }
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

    /// @brief prints out matrix
    void print() {
        grid.print();
    }

};

