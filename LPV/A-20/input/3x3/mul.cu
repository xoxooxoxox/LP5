#include <stdio.h>
#include <cuda.h>

__global__ void arrmul(int *x, int *y, int *z)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    z[id] = x[id] * y[id];
}

int main()
{
    int a[3][3];
    int b[3][3];
    int c[3][3];
    int *d, *e, *f;
    int i, j;

    printf("\nEnter nine elements of the first array:\n");
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            scanf("%d", &a[i][j]);
        }
    }

    printf("\nEnter nine elements of the second array:\n");
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            scanf("%d", &b[i][j]);
        }
    }

    cudaMalloc((void **)&d, 9 * sizeof(int));
    cudaMalloc((void **)&e, 9 * sizeof(int));
    cudaMalloc((void **)&f, 9 * sizeof(int));

    cudaMemcpy(d, a, 9 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, b, 9 * sizeof(int), cudaMemcpyHostToDevice);

    arrmul<<<1, 9>>>(d, e, f);

    cudaMemcpy(c, f, 9 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMultiplication of two arrays:\n");
    for (i = 0; i < 3; i++)
    {
        for (j = 0; j < 3; j++)
        {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}
