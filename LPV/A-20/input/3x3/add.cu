#include <stdio.h>
#include <cuda.h>

__global__ void matAdd(int *a, int *b, int *c)
{
    int row = blockIdx.y;
    int col = blockIdx.x;
    int index = row * 3 + col;
    c[index] = a[index] + b[index];
}

int main()
{
    int a[3][3];
    int b[3][3];
    int c[3][3];
    int *d_a, *d_b, *d_c;
    int i, j;

    printf("Enter nine elements of the first matrix:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            scanf("%d", &a[i][j]);
        }
    }

    printf("\nEnter nine elements of the second matrix:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            scanf("%d", &b[i][j]);
        }
    }

    cudaMalloc((void **)&d_a, 9 * sizeof(int));
    cudaMalloc((void **)&d_b, 9 * sizeof(int));
    cudaMalloc((void **)&d_c, 9 * sizeof(int));

    cudaMemcpy(d_a, a, 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 9 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimGrid(3, 3);
    matAdd<<<dimGrid, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, 9 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSum of two matrices:\n");
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
