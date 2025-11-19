#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__global__ void matmul_tiled(int *A, int *B, int *C, int N, int tile_width)
{
    extern __shared__ int shared_mem[];

    int *mA = shared_mem;
    int *mB = &shared_mem[tile_width * tile_width];
    int row = threadIdx.y + blockIdx.y * tile_width;
    int col = threadIdx.x + blockIdx.x * tile_width;
    int val = 0;

    for (int i = 0; i < (N - 1) / tile_width + 1; i++)
    {
        mA[threadIdx.y * tile_width + threadIdx.x] = (row < N && (i * tile_width + threadIdx.x) < N) ? A[row * N + i * tile_width + threadIdx.x] : 0;
        mB[threadIdx.y * tile_width + threadIdx.x] = (col < N && (i * tile_width + threadIdx.y) < N) ? B[(i * tile_width + threadIdx.y) * N + col] : 0;
        __syncthreads();

        for (int j = 0; j < tile_width; j++)
        {
            val += mA[threadIdx.y * tile_width + j] * mB[j * tile_width + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = val;
    }
}

vector<int> loadMatrix(const string &filename, int &size)
{
    ifstream file(filename);
    vector<int> matrix;
    int value;

    while (file >> value)
    {
        matrix.push_back(value);
        if (file.peek() == ',')
            file.ignore();
    }

    size = sqrt(matrix.size());
    return matrix;
}

int main(int argc, char **argv)
{
    int tile_size = atoi(argv[1]), size;
    vector<int> h_A = loadMatrix(argv[2], size), h_B = loadMatrix(argv[3], size), h_C(size * size);
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
    tile_size = min(tile_size, 32);
    dim3 blockSize(tile_size, tile_size);
    dim3 gridSize((size - 1) / tile_size + 1, (size - 1) / tile_size + 1);
    
    cudaEventRecord(start);
    matmul_tiled<<<gridSize, blockSize, 2 * tile_size * tile_size * sizeof(int)>>>(d_A, d_B, d_C, size, tile_size);
    cudaEventRecord(stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float microseconds = milliseconds * 1000;

    ofstream outfile("output_2_number.csv");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size - 1; j++)
        {
            outfile << h_C[i * size + j] << ",";
        }
        outfile << h_C[i * size + size - 1] << endl;
    }
    outfile.close();

    ofstream resultFile("output_2_number.txt");
    resultFile << size << endl;
    resultFile << (int)microseconds << endl;
    resultFile.close();

    cout << "Product Matrix of size " << size << " stored as output_2_number.csv" << endl;
    cout << "Kernel execution time: " << (int)microseconds << " microseconds" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
