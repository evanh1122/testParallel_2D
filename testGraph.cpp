// mpic++ testGraph.cpp -o testGraph -l armadillo

#include "Grid.cpp"


int main() {
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
    std::pair<double, double> width1 = std::make_pair(1, 3.75);
    std::pair<double, double> height1 = std::make_pair(1, 3.75);
    double interval1 = 0.25;

    Grid grid1(width1, height1, interval1, iProc, nProcs);


    std::pair<double, double> width2 = std::make_pair(0, 5);
    std::pair<double, double> height2 = std::make_pair(0, 5);
    double interval2 = 1;

    Grid grid2(width2, height2, interval2, iProc, nProcs);

    grid1.initCoefficients(&grid2);
    grid2.initCoefficients(&grid1);

    MPI_Barrier(MPI_COMM_WORLD);


    // goes through grid1 10 times
    for (int i = 0; i < 10; ++i) {
        for (double r = height1.first; r <= height1.second; r += interval1) {
            for (double c = width1.first; c <= width1.second; c += interval1) {
                
                double value;
                int proc = grid1.getValue(std::make_pair(r, c), &grid2, &value);
                MPI_Barrier(MPI_COMM_WORLD);
                if (iProc == proc) grid1.setValue(std::make_pair(r, c), value);
            }
        }

        grid1.randomFill();
    }


    MPI_Finalize();
}