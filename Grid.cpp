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
    std::map<double, double> xPos, yPos, xCoeff, yCoeff;
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

    // TO DO: maybe add functionality for different intervals for x and y... should be very simple to do
    // NOTE:  my implementation (with xPos, yPos) does not work if the grid is rotated.

    /// @brief generates a grid filled with random values. Automatically splits up the grid to each processor 
    ///        according to the diagram above
    /// @param width the bounds of the x axis of the grid
    /// @param height the bounds of the y axis of the grid
    /// @param interval the interval of the x and y values
    /// @param iProc the rank of the processor that is running the function
    /// @param nProcs the total number of processors
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


    /// @brief find the coefficients for the local grid. Coefficients are used for interpolation
    ///        Coefficients are always with respect to the smaller x/y-value
    /// @param g the grid that you are using to calculate coefficients for the local grid 
    void initCoefficients(Grid *g) {

        // iterator through all of the x-values in the local grid
        std::map<double, std::vector<int>>::iterator set = this->xOwnership.begin();
        while (set != this->xOwnership.end()) {

            // if 
            if (set->first < g->xOwnership.begin()->first || set->first > g->xOwnership.rbegin()->first) {
                xCoeff[set->first] = -1;
                ++set;
                continue;
            }

            std::map<double, std::vector<int>>::iterator upper;
            upper = g->xOwnership.lower_bound(set->first);

            if (upper->first == set->first) {
                xCoeff[set->first] = 0;
                ++set;
                continue;
            }

            else if (upper != g->xOwnership.end()) {

                std::map<double, std::vector<int>>::iterator lower;
                lower = upper;
                --lower;

                double remainder = set->first - lower->first;
                double interval = upper->first - lower->first;
                xCoeff[set->first] = remainder / interval;
                ++set;
            }
        }

        // do the same thing with the y axis
        set = this->yOwnership.begin();
        while (set != this->yOwnership.end()) {

            if (set->first < g->yOwnership.begin()->first || set->first > g->yOwnership.rbegin()->first) {
                yCoeff[set->first] = -1;
                ++set;
                continue;
            }

            std::map<double, std::vector<int>>::iterator upper;
            upper = g->yOwnership.lower_bound(set->first);

            if (upper->first == set->first) {
                yCoeff[set->first] = 0;
                ++set;
                continue;
            }

            else if (upper != g->yOwnership.end()) {

                std::map<double, std::vector<int>>::iterator lower;
                lower = upper;
                --lower;

                double remainder = set->first - lower->first;
                double interval = upper->first - lower->first;
                yCoeff[set->first] = remainder / interval;
                ++set;
            }
        }
    }


    //IDEA - for interpolation, take the weighted average in the x-direction first, then use those weighted averages to 
    //       calculated the weighted average in the y-direction (or vice versa) to get the overall weighted average
    /*void getValue(std::pair<double, double> pos, int iProc, int proc) {
        double xC = xCoeff.at(pos.first);
        double xY = xCoeff.at(pos.second);
        std::vector<int> xProc = xOwnership[pos.first];
        std::vector<int> yProc = yOwnership[pos.second];

        if (std::find(xProc.begin(), xProc.end(), iProc) != xProc.end() && 
            std::find(yProc.begin(), yProc.end(), iProc) != yProc.end()) {

            double value = grid.at(xPos[pos.first], yPos[pos.second]);
            MPI_Send(&value, 1, MPI_DOUBLE, __, 0, MPI_COMM_WORLD);
        }
    }*/


    /// @brief prints out matrix
    void print() {
        grid.print();
    }
    

    /// @brief prints out the xOwnership and yOwnership maps
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


    void printCoefficients() {

        std::cout << "\nX COEFFICIENTS" << std::endl;
        std::map<double, double>::iterator it = xCoeff.begin();

        while(it != xCoeff.end()) {
            std::cout << "x = " << it->first << ", coeff = " << it->second << std::endl;
            ++it;
        }

        std::cout << "\nY COEFFICIENTS" << std::endl;
        it = yCoeff.begin();

        while (it != yCoeff.end()) {
            std::cout << "y = " << it->first << ", coeff = " << it->second << std::endl;
            ++it;
        } 
    }


    /// @brief out a vector separated by spaces (used in printOwnership)
    /// @param vect the vector you want to print out
    void printVector(std::vector<int> *vect) {

        for (auto x : *vect) {
            std::cout << x << " ";
        }
    }


    /// @brief prints out the xPos and yPos map
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