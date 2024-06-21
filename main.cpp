// mpic++ main.cpp -o main -larmadillo

#include "functions.cpp"


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

    // INCLUSIVE - set the size of the grids
    std::pair<double, double> width1 = std::make_pair(0, 10);
    std::pair<double, double> height1 = std::make_pair(0, 10);

    double interval1 = 1;


    arma::mat grid1 = genGrid(width1, height1, interval1, iProc, nProcs);

    std::cout << "processor: " << iProc << ", ";
    grid1.print("grid1:");

    MPI_Finalize();
    return 0;
}