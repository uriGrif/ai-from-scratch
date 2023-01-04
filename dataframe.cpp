#include "./headers/dataframe.h"

DataFrame::DataFrame() {}

DataFrame::DataFrame(const char *filename, const char &delimiter)
{
    load_csv(filename, delimiter);
}

DataFrame::~DataFrame() {}

void DataFrame::load_csv(const char *filename, const char &delimiter)
{
    std::cout << "Loading csv..." << std::endl;

    std::ifstream file;
    file.open(filename);

    bool is_header = true;
    std::string line;
    std::vector<std::variant<double, std::string>> row;
    while (std::getline(file, line))
    {
        row.clear();
        std::stringstream str(line);

        line.clear();

        while (std::getline(str, line, delimiter))
        {
            if (is_header)
            {
                addColumn(line);
            }
            else
            {
                try
                {
                    row.push_back(std::stod(line));
                }
                catch (const std::exception &e)
                {
                    std::cout << "couldn't convert to double" << std::endl;
                    row.push_back(line);
                }
            }
        }

        if (!is_header) addRow(row);
        
        is_header = false;

    }

    file.close();

    std::cout << "File Loaded!" << std::endl;
}

void DataFrame::print()
{
    for (int i = 0; i < width; i++)
    {
        std::cout << columns[i] << "   ";
    }
    std::cout << std::endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // std::cout << std::get<std::string>(rows[i][j]) << "   ";
            std::visit([](const auto &elem)
                       { std::cout << elem << "   "; },
                       rows[i][j]);
        }
        std::cout << std::endl;
    }
}

int DataFrame::getWidth()
{
    return width;
}

int DataFrame::getHeight()
{
    return height;
}

std::vector<std::variant<double, std::string>> DataFrame::getRow(int index)
{
    return rows[index];
}

std::vector<std::variant<double, std::string>> DataFrame::getCol(std::string column)
{
    int i = 0;
    while (strcmp(columns[i].c_str(), column.c_str()) != 0 && i < columns.size())
    {
        i++;
    }
    if (i == columns.size())
    {
        std::cout << "Column not found" << std::endl;
        exit(1);
    }

    std::vector<std::variant<double, std::string>> col;

    for (int j = 1; j < height; j++)
    {
        col.push_back(rows[j][i]);
    }

    return col;
}

void DataFrame::addColumn(std::string column)
{
    columns.push_back(column);
    width++;
}

void DataFrame::addRow(std::vector<std::variant<double, std::string>> row)
{
    rows.push_back(row);
    height++;
}

void DataFrame::getSimpleMatrixes(double **&train_x,
            double *&train_y,
            double **&test_x,
            double *&test_y,
            int &train_height,
            int &test_height,
            float train_fraction,
            int y_index)
{
    train_height = (height * train_fraction);
    test_height = height - train_height;

    train_x = new double *[train_height];
    train_y = new double[train_height];
    test_x = new double *[test_height];
    test_y = new double[test_height];

    int train_test_limit = (height * train_fraction) - 1;
    int aux;

    for (int i = 0; i < height; i++)
    {
        aux = 0;
        for (int j = 0; j < width; j++) {
            if (i <= train_test_limit)
            {
                if (!j) train_x[i] = new double[width - 1];
                if (j != y_index) {
                    train_x[i][aux] = std::get<double>(rows[i][j]);
                    aux++;
                }
            }
            else
            {
                if (!j) test_x[i - train_height] = new double[width - 1];
                if (j != y_index) {
                    test_x[i - train_height][aux] = std::get<double>(rows[i][j]);
                    aux++;
                }
            }
        }
        if (i <= train_test_limit)
        {
            train_y[i] = std::get<double>(rows[i][y_index]);
        }
        else
        {
            test_y[i - train_height] = std::get<double>(rows[i][y_index]);
        }
        std::vector<std::variant<double, std::string>>().swap(rows[i]); // free memory
    }

    std::vector<std::vector<std::variant<double, std::string>>>().swap(rows); // free memory
}