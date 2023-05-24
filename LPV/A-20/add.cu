#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int N = 3;  // Number of digits in the numbers
const int NumPairs = 10;  // Number of pairs of random numbers

__global__ void addLargeNumbers(int* A, int* B, int* C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumPairs)
    {
        int carry = 0;
        for (int i = N - 1; i >= 0; i--)
        {
            int sum = A[idx * N + i] + B[idx * N + i] + carry;
            C[idx * N + i] = sum % 10;
            carry = sum / 10;
        }
    }
}

int main()
{
    // Generate random numbers
    srand(time(NULL));

    int A[NumPairs][N];
    int B[NumPairs][N];
    int C[NumPairs][N];

    // Generate random numbers for each pair
    for (int i = 0; i < NumPairs; i++)
    {
        int numberA = rand() % 900 + 100;  // Generate a random 3-digit number
        int numberB = rand() % 900 + 100;  // Generate a random 3-digit number

        for (int j = N - 1; j >= 0; j--)
        {
            A[i][j] = numberA % 10;  // Extract the digit
            B[i][j] = numberB % 10;  // Extract the digit
            numberA /= 10;
            numberB /= 10;
        }
    }

    // Allocate memory on the device
    int* dev_A, * dev_B, * dev_C;
    cudaMalloc(&dev_A, NumPairs * N * sizeof(int));
    cudaMalloc(&dev_B, NumPairs * N * sizeof(int));
    cudaMalloc(&dev_C, NumPairs * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_A, A, NumPairs * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, NumPairs * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (NumPairs + blockSize - 1) / blockSize;
    addLargeNumbers<<<numBlocks, blockSize>>>(dev_A, dev_B, dev_C);

    // Copy data from device to host
    cudaMemcpy(C, dev_C, NumPairs * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < NumPairs; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << A[i][j];
        }
        std::cout << " + ";
        for (int j = 0; j < N; j++)
        {
            std::cout << B[i][j];
        }
        std::cout << " = ";
        for (int j = 0; j < N; j++)
        {
            std::cout << C[i][j];
        }
        std::cout << std::endl;
    }

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}
