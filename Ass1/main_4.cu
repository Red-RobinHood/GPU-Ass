#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void mattrans_tiled(int* A, int* C, int N, int tile_width) {
    extern __shared__ int tile[];
    
    int blockRow = blockIdx.y * tile_width;
    int blockCol = blockIdx.x * tile_width;
    int row = blockRow + threadIdx.y;
    int col = blockCol + threadIdx.x;

    if (row < N && col < N) {
        tile[threadIdx.y * tile_width + threadIdx.x] = A[row * N + col];
    }
    __syncthreads();

    row = blockCol + threadIdx.y;
    col = blockRow + threadIdx.x;

    if(row < N && col < N) {
        C[row * N + col] = tile[threadIdx.x * tile_width + threadIdx.y];
    }
}

vector<int> loadMatrix(const string& filename, int& size) {
    ifstream file(filename);
    vector<int> matrix;
    int value;

    while (file >> value) {
        matrix.push_back(value);
        if (file.peek() == ',') 
            file.ignore();
    }

    size = sqrt(matrix.size());
    return matrix;
}

int main(int argc, char** argv) {
    int tile_size = atoi(argv[1]), size;
    vector<int> h_A = loadMatrix(argv[2], size), h_C(size * size);
    int bytes = sizeof(int) * size * size;
    int *d_A, *d_C;
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    tile_size = min(tile_size, 32);
    dim3 blockSize(tile_size, tile_size);
    dim3 gridSize((size - 1) / tile_size + 1, (size - 1) / tile_size + 1);

    cudaEventRecord(start);
    mattrans_tiled<<<gridSize, blockSize, tile_size * (tile_size + 1) * sizeof(int)>>>(d_A, d_C, size, tile_size);
    cudaEventRecord(stop);
    
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float microseconds = milliseconds * 1000;

    ofstream outfile("output_4_CS23BTECH11036.csv");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++) {
            outfile << h_C[i * size + j] << ",";
        }
        outfile << h_C [i * size + size - 1] << endl;
    }
    outfile.close();

    ofstream resultFile("output_4_CS23BTECH11036.txt");
    resultFile << size << endl;
    resultFile << (int)microseconds << endl;
    resultFile.close();
    
    cout << "Product Matrix of size " << size << " stored as output_4_CS23BTECH11036.csv" << endl;
    cout << "Kernel execution time: " << (int)microseconds << " microseconds" << endl;
    
    cudaFree(d_A);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
