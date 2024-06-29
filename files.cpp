#include <fstream>
#include <string>

void removeEmptyLines(const std::string &file) {
    std::ifstream fin(file);
    std::string line, text;
    
    while (std::getline(fin, line)) {
        if (!(line.empty() || line.find_first_not_of(' ') == std::string::npos)) {
            text += line + "\n";
        }
    }

    fin.close();
    std::ofstream out(file);
    out << text;
}
