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
    std::map<double, double> xPos, yPos;
    std::map<double, std::vector<int>> xOwnership, yOwnership;
    std::pair<double, double> width, height;
    double interval;
    int iProc, nProcs, root_nProcs, subWidth, subHeight;

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
    Grid(std::pair<double, double> _width, std::pair<double, double> _height, double _interval, int _iProc, 
         int _nProcs) : width(_width), height(_height), interval(_interval), nProcs(_nProcs), iProc(_iProc) {

        // CREATING THE GRIDS
        // calculates the width and height of a matrix that each processor is responsible for
        // adjusts for if width and/or height isn't perfectly divisible by the number of processors
        root_nProcs = sqrt(nProcs);
        subWidth = (abs(width.second - width.first + interval) / interval) / root_nProcs;
        if (fmod(abs(width.second - width.first + interval) / interval, root_nProcs) != 0) ++subWidth;

        subHeight = (abs(height.second - height.first + interval) / interval) / root_nProcs;
        if (fmod(abs(height.second - height.first + interval) / interval, root_nProcs) != 0) ++subHeight;


        // if processor is responsible for an edge grid, then make sure that the grid doesn't go too far
        if (iProc % root_nProcs == root_nProcs - 1) 
            subWidth = ((width.second + interval) / interval) - (subWidth * (root_nProcs - 1));

        if (iProc / root_nProcs == root_nProcs - 1) 
            subHeight = ((height.second + interval) / interval) - (subHeight * (root_nProcs - 1));


        // creates the grid on each processor with random numbers
        MPI_Barrier(MPI_COMM_WORLD);
        createGrid();

        // OWNERSHIP MAPS
        initOwnership();

    }


    /// @brief initializes the grid for every processor
    void createGrid() {
        grid.set_size(subHeight, subWidth);
        grid.fill(arma::fill::randu);
    }


    /// @brief initializes the x and y ownership maps
    void initOwnership() {
        // fill the ownership maps up with empty vectors
        for (double i = width.first; i <= width.second; i += interval) xOwnership[i] = std::vector<int>();
        for (double i = height.first; i <= height.second; i += interval) yOwnership[i] = std::vector<int>();

        // honestly I don't know how to really explain this or how I even figured this out...
        // iteration is used for calculation of index values for xPos and yPos maps
        int iteration = 0;
        for (int c = 0; c < root_nProcs; ++c) {

            for (double x = width.first + (c * subWidth * interval); x < width.first + ((c + 1) * subWidth * interval) &&
                 x <= width.second; x += interval) {

                for (int proc = 0; proc < nProcs ; ++proc) {

                    if (proc % root_nProcs == c) {
                        xOwnership.at(x).push_back(proc);

                        if (iProc == proc) {
                            // initializing xPos and filling every proc with their respective values
                            xPos[x] = iteration - (c * subWidth);
                        }
                    }
                }
                ++iteration;
            }
        }

        // do the same thing for yOwnership
        iteration = 0;
        for (int r = 0; r < root_nProcs; ++r) {

            for (double y = height.first + (r * subHeight * interval); y < height.first + ((r + 1) * subHeight * interval) &&
                 y <= height.second; y += interval) {
                
                for (int proc = 0; proc < nProcs ; ++proc) {

                    if (proc / root_nProcs == r) {
                        yOwnership.at(y).push_back(proc);

                        if (iProc == proc) yPos[y] = iteration - (r * subHeight);
                    }
                }
                ++iteration;
            }
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

    void printXandY() {
        std::cout << "\nprocessor " << iProc << std::endl;
        
        std::map<double, double>::iterator it = xPos.begin();
        while (it != xPos.end()) {
            std::cout << "x = " << it->first << ", index = " << it->second << std::endl;
            ++it;
        }

        it = yPos.begin();
        std:: cout << std::endl;
        while (it != yPos.end()) {
            std::cout << "y = " << it->first << ", index = " << it->second << std::endl;
            ++it;
        }
    }

};

