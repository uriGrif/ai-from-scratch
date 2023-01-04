#pragma once

#include <iostream>
#include <vector>
#include <string.h>
#include <variant>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>

class DataFrame {
    private:
        int width = 0;
        int height = 0;

        std::vector<std::string> columns;

        std::vector<std::vector<std::variant<double, std::string>>> rows;

    public:
        DataFrame();
        DataFrame(const char *filename, const char &delimiter=',');
        ~DataFrame();
        void print();
        void load_csv(const char *filename, const char &delimiter=',');
        int getWidth();
        int getHeight();
        std::vector<std::variant<double, std::string>> getRow(int index);
        std::vector<std::variant<double, std::string>> getCol(std::string column);
        void addColumn(std::string column);
        void addRow(std::vector<std::variant<double, std::string>> row);
        void getSimpleMatrixes(
            double **&train_x,
            double *&train_y,
            double **&test_x,
            double *&test_y,
            int &train_height,
            int &test_height,
            float train_fraction=0.75f,
            int y_index=0);
};