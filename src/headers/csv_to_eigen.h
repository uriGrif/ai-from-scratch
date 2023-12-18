#ifndef CSV_TO_EIGEN_INCLUDED
#define CSV_TO_EIGEN_INCLUDED

#include <Eigen/Dense>
#include <vector>
#include <fstream>

using namespace Eigen;

template <typename M>
M load_csv(const std::string &path, bool hasHeader = false)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    int rows = 0;
    if (hasHeader)
        std::getline(indata, line);
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size() / rows);
}

#endif