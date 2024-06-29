// mpic++ testGraph.cpp -o testGraph -l armadillo

#include "Grid.cpp"
#include "files.cpp"


int main() {
    int iProc, nProcs;
    
    // initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &iProc);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

    // make sure that the program is ran with a perfect square amount of processors (1, 4, 9, 16, etc.)
    /*
    if (floor(sqrt(nProcs)) != ceil(sqrt(nProcs))) {
        throw std::runtime_error("ERROR: Please run with a perfect square amount of processors (1, 4, 9, 16, etc.)");
    }
    */
    if (nProcs != 4) {
        throw std::runtime_error("ERROR: Program currently only works with 4 processors");
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


    
    std::ofstream fout;
    std::string file = "data";
    file += std::to_string(iProc);
    file += ".txt";

    int proc;
    for (int i = 0; i < 1; ++i) {

        fout.open(file, std::ios::out | std::ios::trunc);

        for (double r = height1.first; r <= height1.second; r += interval1) {
            for (double c = width1.first; c <= width1.second; c += interval1) {
                
                double value;
                proc = grid1.getValue(std::make_pair(r, c), &grid2, &value);
                MPI_Barrier(MPI_COMM_WORLD);
                if (iProc == proc) {
                    grid1.setValue(std::make_pair(r, c), value);
                    fout << value << " ";
                }
            }

            fout << "\n";
        }

        fout.close();
        MPI_Barrier(MPI_COMM_WORLD);
        removeEmptyLines(file);
        MPI_Barrier(MPI_COMM_WORLD);

        // combines all of the text files onto data.txt in the format of the grid
        // currently only works for when there are 4 processors for simplicity reasons
        if (iProc == 0) {
            fout.open("data.txt", std::ios::out | std::ios::trunc);

            std::ifstream fin0("data0.txt", std::ios::in);
            std::ifstream fin1("data1.txt", std::ios::in);
            std::string line0, line1;

            while (std::getline(fin0, line0)) {
                std::getline(fin1, line1);
                std::string text = line0 + line1;
                fout << text << "\n";
            }

            fin0.close();
            fin1.close();

            std::ifstream fin2("data2.txt", std::ios::in);
            std::ifstream fin3("data3.txt", std::ios::in);
            std::string line2, line3;

            while (std::getline(fin2, line2)) {
                std::getline(fin3, line3);
                std::string text = line2 + line3;
                fout << text << "\n";
            }

            fin2.close();
            fin3.close();
            fout.close();
        }

        //grid1.randomFill();
    }


    MPI_Finalize();
}