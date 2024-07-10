// mpic++ testGraph.cpp -o testGraph -l armadillo

#include "Grid.cpp"
#include "files.cpp"


// NOTE - this currently spits out the grid for all of grid1 and grid2
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
    std::pair<double, double> width1 = std::make_pair(0, 24.75);
    std::pair<double, double> height1 = std::make_pair(0, 24.75);
    double intervalX1 = 0.25;
    double intervalY1 = 0.25;

    Grid grid1(width1, height1, intervalX1, intervalY1, iProc, nProcs);
    grid1.initGridRand();


    std::pair<double, double> width2 = std::make_pair(0, 25);
    std::pair<double, double> height2 = std::make_pair(0, 25);
    double intervalX2 = 1;
    double intervalY2 = 1;

    Grid grid2(width2, height2, intervalX2, intervalY2, iProc, nProcs);
    grid2.initGridSin();

    grid1.initCoefficients(&grid2);
    grid2.initCoefficients(&grid1);

    MPI_Barrier(MPI_COMM_WORLD);


    
    std::ofstream fout1, fout2;
    std::string file1 = "/home/evanh1122/testParallel_2D/output/dataCopy";
    file1 += std::to_string(iProc);
    file1 += ".txt";

    std::string file2 = "/home/evanh1122/testParallel_2D/output/dataOriginal";
    file2 += std::to_string(iProc);
    file2 += ".txt";


    int proc;
    for (int i = 0; i < 1; ++i) {

        fout1.open(file1, std::ios::out | std::ios::trunc);

        for (double r = height1.first; r <= height1.second; r += intervalY1) {
            for (double c = width1.first; c <= width1.second; c += intervalX1) {
                
                double value;
                proc = grid1.getValue(std::make_pair(c, r), &grid2, &value);
                MPI_Barrier(MPI_COMM_WORLD);
                if (iProc == proc) {
                    grid1.setValue(std::make_pair(c, r), value);
                    fout1 << value << " ";
                }
            }

            fout1 << std::endl;
        }
        fout1.close();


        // do the same thing for the original grid
        fout2.open(file2, std::ios::out | std::ios::trunc);

        for (double r = height2.first; r <= height2.second; r += intervalY2) {
            for (double c = width2.first; c <= width2.second; c += intervalX2) {
                
                double value;
                if (grid2.contains(std::make_pair(c, r))) {
                    proc = grid2.getValue(std::make_pair(c, r), &value);
                    if (iProc == proc) {
                        fout2 << value << " ";
                    }
                }
            }

            fout2 << std::endl;
        }
        fout2.close();


        MPI_Barrier(MPI_COMM_WORLD);
        removeEmptyLines(file1);
        removeEmptyLines(file2);
        MPI_Barrier(MPI_COMM_WORLD);

        // combines all of the text files onto data.txt in the format of the grid
        // currently only works for when there are 4 processors for simplicity reasons
        if (iProc == 0) {
            fout1.open("dataCopy.txt", std::ios::out | std::ios::trunc);

            std::ifstream fin0("/home/evanh1122/testParallel_2D/output/dataCopy0.txt", std::ios::in);
            std::ifstream fin1("/home/evanh1122/testParallel_2D/output/dataCopy1.txt", std::ios::in);
            std::string line0, line1;

            while (std::getline(fin0, line0)) {
                std::getline(fin1, line1);
                std::string text = line0 + line1;
                fout1 << text << std::endl;
            }

            fin0.close();
            fin1.close();

            std::ifstream fin2("/home/evanh1122/testParallel_2D/output/dataCopy2.txt", std::ios::in);
            std::ifstream fin3("/home/evanh1122/testParallel_2D/output/dataCopy3.txt", std::ios::in);
            std::string line2, line3;

            while (std::getline(fin2, line2)) {
                std::getline(fin3, line3);
                std::string text = line2 + line3;
                fout1 << text << std::endl;
            }

            fin2.close();
            fin3.close();
            fout1.close();
        }

        
        // do the same thing for the original grid data
        if (iProc == 0) {
            fout2.open("dataOriginal.txt", std::ios::out | std::ios::trunc);

            std::ifstream fin0("/home/evanh1122/testParallel_2D/output/dataOriginal0.txt", std::ios::in);
            std::ifstream fin1("/home/evanh1122/testParallel_2D/output/dataOriginal1.txt", std::ios::in);
            std::string line0, line1;

            while (std::getline(fin0, line0)) {
                std::getline(fin1, line1);
                std::string text = line0 + line1;
                fout2 << text << std::endl;
            }

            fin0.close();
            fin1.close();

            std::ifstream fin2("/home/evanh1122/testParallel_2D/output/dataOriginal2.txt", std::ios::in);
            std::ifstream fin3("/home/evanh1122/testParallel_2D/output/dataOriginal3.txt", std::ios::in);
            std::string line2, line3;

            while (std::getline(fin2, line2)) {
                std::getline(fin3, line3);
                std::string text = line2 + line3;
                fout2 << text << std::endl;
            }

            fin2.close();
            fin3.close();
            fout2.close();
        }
        

        //grid1.randomFill();
    }


    MPI_Finalize();
}