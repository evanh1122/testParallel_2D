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
    std::map<double, double> x, y;
    std::map<double, std::vector<int>> xOwnership, yOwnership;
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
    // TO DO: maybe add functionality for different intervals for x and y... should be very simple to do
    Grid(std::pair<double, double> _width, std::pair<double, double> _height, double _interval, int iProc, 
         int _nProcs) : width(_width), height(_height), interval(_interval), nProcs(_nProcs) {

        // CREATING THE GRIDS
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


        // creates the grid on each processor with random numbers
        grid.set_size(subHeight, subWidth);
        grid.fill(arma::fill::randu);


        // OWNERSHIP MAPS
        // fill the ownership maps up with empty vectors
        for (double i = width.first; i <= width.second; i += interval) xOwnership[i] = std::vector<int>();
        for (double i = height.first; i <= height.second; i += interval) yOwnership[i] = std::vector<int>();

        // honestly I don't know how to really explain this or how I even figured this out...
        for (int c = 0; c < root_nProcs; ++c) {

            for (double x = width.first + (c * subWidth * interval); x < width.first + ((c + 1) * subWidth * interval) &&
                 x <= width.second; x += interval) {

                for (int proc = 0; proc < nProcs ; ++proc) {

                    if (proc % root_nProcs == c) xOwnership.at(x).push_back(proc);
                }
            }
        }

        for (int r = 0; r < root_nProcs; ++r) {

            for (double y = height.first + (r * subHeight * interval); y < height.first + ((r + 1) * subHeight * interval) &&
                 y <= height.second; y += interval) {
                
                for (int proc = 0; proc < nProcs ; ++proc) {

                    if (proc / root_nProcs == r) yOwnership.at(y).push_back(proc);
                }
            }
        }



        /*int iMat = 0;
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
        }*/
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
    
    void printOwnership() {

        std::cout << "\nX OWNERSHIP" << std::endl;
        std::map<double, std::vector<int>>::iterator it = xOwnership.begin();

        while(it != xOwnership.end()) {

            std::cout << "x = " << it->first << ", procs: ";
            printVector(&it->second);
            std::cout << std::endl;
            ++it;
        }

        std::cout << "\nY OWNERSHIP" << std::endl;
        it = yOwnership.begin();

        while (it != yOwnership.end()) {

            std::cout << "y = " << it->first << ", procs: ";
            printVector(&it->second);
            std::cout << std::endl;
            ++it;
        }
    }

    void printVector(std::vector<int> *vect) {

        for (auto x : *vect) {
            std::cout << x << " ";
        }
    }

};

