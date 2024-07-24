/*
 This is an example code that shows interpolation between two grids.
 This code allows communication between multiple processors (using openMPI)
 so that one grid can get data from another grid (despite being on the 
 data being on separate processors). If the point we're interested in 
 doesn't exist on the grid we're interested in, then we will estimate
 the value using interpolation (through the use of coefficients)

 This code uses arma::mat and only works in 2D.

 Compile using:
 mpic++ main.cpp -o main -l armadillo

*/

#include "Grid.cpp"


int main () {
    int iProc, nProcs;
    
    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // make sure that the program is ran with a perfect square amount of processors (1, 4, 9, 16, etc.)
    if (floor(sqrt(nProcs)) != ceil(sqrt(nProcs))) {
        throw std::runtime_error("ERROR: Please run with a perfect square amount of processors (1, 4, 9, 16, etc.)");
    }


    // [double, double] - set the bounds of the grids (inclusive lower and upper bounds)
    std::pair<double, double> width1 = std::make_pair(-1, 2.75);
    std::pair<double, double> height1 = std::make_pair(-1, 2.75);

    // set the intervals for the x and y values
    double intervalX1 = 0.25;
    double intervalY1 = 0.25;


    // create a grid and fill it with random data
    Grid grid1(width1, height1, intervalX1, intervalY1, iProc, nProcs);
    grid1.initGridRand();

    // prints out the sub-grid that each processor is responsible for
    // NOTE - due to nature of MPI, you may need to run multiple times for the output to lineup properly
    std::cout << "processor: " << iProc << std::endl;
    grid1.print();

    // prints out what
    sleep(0.9);
    grid1.printOwnership();

    MPI_Barrier(MPI_COMM_WORLD);
    sleep(0.9);
    grid1.printXandY();


    std::pair<double, double> width2 = std::make_pair(-1, 4);
    std::pair<double, double> height2 = std::make_pair(-1, 4);

    double intervalX2 = 1;
    double intervalY2 = 1;

    Grid grid2(width2, height2, intervalX2, intervalY2, iProc, nProcs);
    grid2.initGridRand();

    MPI_Barrier(MPI_COMM_WORLD);
    grid1.initCoefficients(&grid2);
    grid2.initCoefficients(&grid1);

    if (iProc == 0) grid1.printCoefficients();

    MPI_Barrier(MPI_COMM_WORLD);
    if (iProc == 0) std::cout << "\nGRID 2:" << std::endl;
    sleep(0.9);
    std::cout << "processor: " << iProc << std::endl;
    grid2.print();


    MPI_Barrier(MPI_COMM_WORLD);
    sleep(0.9);
    if (iProc == 0) std::cout << "\nTESTING SEND AND RECV" << std::endl;

    // change this to test getting the data at different positions
    std::pair<double, double> pos = std::make_pair(-1, -1);

    double test;
    int proc = grid1.getValue(pos, &grid2, &test);
    if (iProc == proc) {
        std::cout << "proc: " << iProc << ", value = " << test << std::endl;
    }


    MPI_Finalize();
    return 0;
}