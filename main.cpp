// mpic++ main.cpp -o main -l armadillo

#include "Grid.cpp"


int main () {
    int iProc, nProcs;
    
    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // make sure that the program is ran with a perfect square amount of processors (1, 4, 9, 16, etc.)
    if (floor(sqrt(nProcs)) != ceil(sqrt(nProcs))) {
        std::cout << "ERROR: Please run with a perfect square amount of processors (1, 4, 9, 16, etc.)" << std::endl;
        return -1;
    }

    // sets a random seed for each processor to generate random numbers to fill the grid
    arma::arma_rng::set_seed_random();

    // [double, double] - set the size of the grids (inclusive lower and upper bounds)
    // NOTE - negatives currently don't work
    std::pair<double, double> width1 = std::make_pair(0, 2.75);
    std::pair<double, double> height1 = std::make_pair(0, 2.75);

    double interval1 = 0.25;


    Grid grid1(width1, height1, interval1, iProc, nProcs);

    std::cout << "processor: " << iProc << std::endl;
    grid1.print();

    sleep(0.9);
    if (iProc == 0) grid1.printOwnership();

    MPI_Barrier(MPI_COMM_WORLD);
    sleep(0.9);
    grid1.printXandY();


    std::pair<double, double> width2 = std::make_pair(1, 4);
    std::pair<double, double> height2 = std::make_pair(1, 4);

    double interval2 = 1;

    Grid grid2(width2, height2, interval2, iProc, nProcs);

    MPI_Barrier(MPI_COMM_WORLD);
    grid1.initCoefficients(&grid2);

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
    std::pair<double, double> pos = std::make_pair(1.5, 1.5);

    double test;
    int proc = grid1.getValue(pos, &grid2, &test);
    if (iProc == proc) {
        std::cout << "proc: " << iProc << ", value = " << test << std::endl;
    }


    MPI_Finalize();
    return 0;
}