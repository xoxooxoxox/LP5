#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int N = 3;  // Size of the square matrices

__global__ void matrixMultiply(int* A, int* B, int* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main()
{
    // Generate random matrices A and B
    srand(time(NULL));

    int A[N][N];
    int B[N][N];
    int C[N][N];

    // Generate random numbers for matrix A
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i][j] = rand() % 10;  // Generate a random number between 0 and 9
        }
    }

    // Generate random numbers for matrix B
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            B[i][j] = rand() % 10;  // Generate a random number between 0 and 9
        }
    }

    // Allocate memory on the device
    int* dev_A, * dev_B, * dev_C;
    cudaMalloc(&dev_A, N * N * sizeof(int));
    cudaMalloc(&dev_B, N * N * sizeof(int));
    cudaMalloc(&dev_C, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 dimBlock(N, N);
    dim3 dimGrid(1, 1);
    matrixMultiply<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);

    // Copy data from device to host
    cudaMemcpy(C, dev_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the matrices and the result
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
