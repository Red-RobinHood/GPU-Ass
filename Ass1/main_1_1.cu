#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void matmul_1d(int* A, int* B, int* C, int N) {
    int loop = (N * N - 1) / (blockDim.x * gridDim.x) + 1;
    for(int i = 0; i < loop; i++) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x + i * blockDim.x * gridDim.x;
        if (idx < N * N) {
            int row = idx / N;
            int col = idx % N;

            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
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
    int M = atoi(argv[1]), N = atoi(argv[2]), size;
    vector<int> h_A = loadMatrix(argv[3], size), h_B = loadMatrix(argv[4], size), h_C(size * size);
    int bytes = sizeof(int) * size * size;
    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1d<<<M, N>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float microseconds = milliseconds * 1000;
    
    ofstream outfile("output_1_1_CS23BTECH11036.csv");
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size - 1; j++) {
            outfile << h_C[i * size + j] << ",";
        }
        outfile << h_C[i * size + size - 1] << endl;
    }
    outfile.close();
    
    ofstream resultFile("output_1_1_CS23BTECH11036.txt");
    resultFile << size << endl;
    resultFile << (int)microseconds << endl;
    resultFile.close();
    
    cout << "Product Matrix of size " << size << " stored as output_1_1_CS23BTECH11036.csv" << endl;
    cout << "Kernel execution time: " << (int)microseconds << " microseconds" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
