CPP_FLAGS = -g -Wall -I ./
SRC_DIR = ./src/
BUILD_DIR = ./build/
CURRENT_DIR = $(shell pwd)

$(BUILD_DIR)main: $(BUILD_DIR)main.o $(BUILD_DIR)neural-network.o build_dir
	g++ $(CPP_FLAGS) $(BUILD_DIR)main.o $(BUILD_DIR)neural-network.o -o $(BUILD_DIR)main

$(BUILD_DIR)main.o: $(SRC_DIR)main.cpp $(SRC_DIR)headers/neural-network.h build_dir
	g++ $(CPP_FLAGS) -c $(SRC_DIR)main.cpp -o $(BUILD_DIR)main.o

$(BUILD_DIR)neural-network.o: $(SRC_DIR)headers/neural-network.h $(SRC_DIR)headers/csv_to_eigen.h build_dir
	g++ $(CPP_FLAGS) -c $(SRC_DIR)neural-network.cpp -o $(BUILD_DIR)neural-network.o

build_dir:
	$(shell mkdir -p $(BUILD_DIR))

clean: 
	rm -rf directoryname $(BUILD_DIR)
