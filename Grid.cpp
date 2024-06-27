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
            subWidth = ((width.second - width.first + interval) / interval) - (subWidth * (root_nProcs - 1));

        if (iProc / root_nProcs == root_nProcs - 1) 
            subHeight = ((height.second - height.first + interval) / interval) - (subHeight * (root_nProcs - 1));


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

            // if the point we're interested in isn't present on grid g, then let the coefficient be -1
            if (set->first < g->xOwnership.begin()->first || set->first > g->xOwnership.rbegin()->first) {
                xCoeff[set->first] = -1;
                ++set;
                continue;
            }

            // if both grids have the same x-value, then upper will point to that value
            // otherwise, upper wil point to the next thighest x-value on grid g
            std::map<double, std::vector<int>>::iterator upper;
            upper = g->xOwnership.lower_bound(set->first);

            // if both grids have the same x-value, then let the coefficient be 0
            if (upper->first == set->first) {
                xCoeff[set->first] = 0;
                ++set;
                continue;
            }

            // if both grids don't have the same x-value, calculate and set the coefficient
            else if (upper != g->xOwnership.end()) {

                // lower will point to the next smaller x-value
                std::map<double, std::vector<int>>::iterator lower;
                lower = upper;
                --lower;

                // calculates and sets the coefficient (with respect to the smaller x-value)
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


    // NOTE - this function assumes that the local grid calling the function has this point pos on its grid (no interpolation)
    int getValue(std::pair<double, double> pos, Grid *get, double *answer) {

        // makes sure that the local grid has pos already on it (no interpolation needed)
        if (!grid.at(yPos[pos.second], xPos[pos.first])) 
            throw std::runtime_error("getValue: This position doesn't exist on the object calling this function!");


        // finds the processor that needs to receive the data
        int receive = this->findProc(pos);

        // finds the coefficients for x and y at the specified position
        double xC = xCoeff.at(pos.first);
        double yC = yCoeff.at(pos.second);

        // if no interpolation is needed (coefficients for x and y are both zero)
        if (xC == 0 && yC == 0) {

            // finds the processor that needs to send the data
            int send = get->findProc(pos);
            
            get->sendRecv(pos, receive, send, answer);
            return receive;
        }


        // if we only have to interpolate in the y-direction
        else if (xC == 0) {

            // gets the processors responsible for the data right above and right below the specified y-value
            std::pair<std::pair<int, int>, std::pair<int, int>> newY = get->getY_UpperLowerProcs(pos);

            // variables to make things easier to understand
            double yLowerPos = newY.first.first; // the y-position directly below pos.second
            double yUpperPos = newY.first.second; // the y-position directly above pos.second
            int yLowerProc = newY.second.first; // the processor responsible for the data at yLowerPos
            int yUpperProc = newY.second.second; // the processor responsible for the data at yUpperProc
            double upperData, lowerData; // where the upper and lower data will be stored

            // MPI send and recv to get the data to the receive processor
            get->sendRecv(std::make_pair(pos.first, yLowerPos), receive, yLowerProc, &lowerData);
            get->sendRecv(std::make_pair(pos.first, yUpperPos), receive, yUpperProc, &upperData);

            // interpolates (weighted average) the data and stores it in answer
            MPI_Barrier(MPI_COMM_WORLD);
            if (iProc == receive) *answer = (yC * upperData) + ((1 - yC) * lowerData);

            // returns the processor that has the newly interpolated data
            return receive;
        }


        // if we only need to interpolate in the x direction
        else if (yC == 0) {

            // this is the same processor as above, just now for x instead of y
            std::pair<std::pair<int, int>, std::pair<int, int>> newX = get->getX_UpperLowerProcs(pos);

            double xLowerPos = newX.first.first;
            double xUpperPos = newX.first.second;
            int xLowerProc = newX.second.first;
            int xUpperProc = newX.second.second;
            double upperData, lowerData;

            get->sendRecv(std::make_pair(xLowerPos, pos.second), receive, xLowerProc, &lowerData);
            get->sendRecv(std::make_pair(xUpperPos, pos.second), receive, xUpperProc, &upperData);

            MPI_Barrier(MPI_COMM_WORLD);
            if (iProc == receive) *answer = (xC * upperData) + ((1 - xC) * lowerData);

            return receive;
        }


        // if we need to interpolate in both the x and y directions
        else {

            // find the y-values that are directely above and below pos.second
            std::map<double, double>::iterator upperY, lowerY;
            upperY = get->yCoeff.upper_bound(pos.second);
            lowerY = upperY;
            --lowerY;


            // gets the data at the x-values directly above and below pos.first at upperY and calculates the weighted average
            std::pair<std::pair<int, int>, std::pair<int, int>> upperX = get->getX_UpperLowerProcs(std::make_pair(pos.first, upperY->first));

            double upperY_xLowerPos = upperX.first.first;
            double upperY_xUpperPos = upperX.first.second;
            int upperY_xLowerProc = upperX.second.first;
            int upperY_xUpperProc = upperX.second.second;
            double upperY_upperXData, upperY_lowerXData, newUpperYData;

            get->sendRecv(std::make_pair(upperY_xLowerPos, upperY->first), receive, upperY_xLowerProc, &upperY_lowerXData);
            get->sendRecv(std::make_pair(upperY_xUpperPos, upperY->first), receive, upperY_xUpperProc, &upperY_upperXData);

            MPI_Barrier(MPI_COMM_WORLD);
            if (iProc == receive) newUpperYData = (xC * upperY_upperXData) + ((1 - xC) * upperY_lowerXData);


            // gets the data at the x-values directly above and below pos.first at lowerY and calculates the weighted average
            std::pair<std::pair<int, int>, std::pair<int, int>> lowerX = get->getX_UpperLowerProcs(std::make_pair(pos.first, lowerY->first));

            double lowerY_xLowerPos = lowerX.first.first;
            double lowerY_xUpperPos = lowerX.first.second;
            int lowerY_xLowerProc = lowerX.second.first;
            int lowerY_xUpperProc = lowerX.second.second;
            double lowerY_upperXData, lowerY_lowerXData, newLowerYData;

            get->sendRecv(std::make_pair(lowerY_xLowerPos, lowerY->first), receive, lowerY_xLowerProc, &lowerY_lowerXData);
            get->sendRecv(std::make_pair(lowerY_xUpperPos, lowerY->first), receive, lowerY_xUpperProc, &lowerY_upperXData);


            // calculates the weighted average between the previously calculated weighted averages
            MPI_Barrier(MPI_COMM_WORLD);
            if (iProc == receive) {
                newLowerYData = (xC * lowerY_upperXData) + ((1 - xC) * lowerY_lowerXData);
                *answer = (yC * newUpperYData) + ((1 - yC) * newLowerYData);
            }

            // returns the rank of the processor that contains the newly interpolated data
            return receive;
            
        }
    }


    /// @brief sends the data that we want to the processor that needs it
    ///        NOTE - data must already exist at the location (no interpolation)
    /// @param pos the position that we are interested in getting the data from
    /// @param receive the processor rank that needs to receive the data
    /// @param send the processor rank that needs to send the data
    /// @param answer this is where the sent data will be stored on the receive processor
    void sendRecv(std::pair<double, double> pos, int receive, int send, double *answer) {

        // if the same processor contains the data that needs to be sent and received, simply get that data
        if (send == receive) {
            if (iProc == receive) *answer = this->grid.at(yPos[pos.second], xPos[pos.first]);
        }

        // otherwise, use MPI to get the data from the send processor to the receive processor
        else {
            if (iProc == send) {
                double data = grid.at(yPos[pos.second], xPos[pos.first]);
                MPI_Send(&data, 1, MPI_DOUBLE, receive, 1, MPI_COMM_WORLD);
            }
            if (iProc == receive)
                MPI_Recv(answer, 1, MPI_DOUBLE, send, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }


    /// @brief finds the processor that is responsible for the data at position pos
    /// @param pos the position we are interested in
    /// @return returns the processor rank that contains data at position pos
    int findProc(std::pair<double, double> pos) {

        // gets the processors that are responsible for data at the specified x and y positions
        std::vector<int> xProc = xOwnership[pos.first];
        std::vector<int> yProc = yOwnership[pos.second];

        // checks to see if there are any matching processors in xProc and yProc. If there are, return that rank
        for (int proc : xProc) {
            if (std::find(yProc.begin(), yProc.end(), proc) != yProc.end()) 
                return proc;
        }

        // if we reach this point, that means that the position does not already exist on the grid
        throw std::runtime_error("findProc: This position is not within the bounds of the grid!");
    }


    /// @brief gets the processors responsible for the data that's right above and right below pos.first
    ///        (used for interpolation purposes)
    /// @param pos the position that we are interested in
    /// @return pair<(lower x-value, upper x-value), (lower proc, upper proc)>
    std::pair<std::pair<int, int>, std::pair<int, int>> getX_UpperLowerProcs(std::pair<double, double> pos) {

        std::map<double, std::vector<int>>::iterator upper, lower;
        upper = xOwnership.upper_bound(pos.first);
        lower = upper;
        --lower;

        int l = findProc(std::make_pair(lower->first, pos.second));
        int u = findProc(std::make_pair(upper->first, pos.second));
        return std::make_pair(std::make_pair(lower->first, upper->first), std::make_pair(l, u));
    }


    /// @brief gets the processors responsible for the data that's right above and right below pos.second
    ///        (used for interpolation purposes)
    /// @param pos the position that we are interested in
    /// @return pair<lower proc, upper proc>
    std::pair<std::pair<int, int>, std::pair<int, int>> getY_UpperLowerProcs(std::pair<double, double> pos) {

        std::map<double, std::vector<int>>::iterator upper, lower;
        upper = yOwnership.upper_bound(pos.second);
        lower = upper;
        --lower;

        int l = findProc(std::make_pair(pos.first, lower->first));
        int u = findProc(std::make_pair(pos.first, upper->first));
        return std::make_pair(std::make_pair(lower->first, upper->first), std::make_pair(l, u));
    }


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

    /// @brief prints the x and y coefficients for the local grid
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