#include <stdio.h>
#include <cuda.h>

__global__ void arrmul(int *x, int *y, int *z) // kernel definition
{
    int id = blockIdx.x;
    /* blockIdx.x gives the respective block id which starts from 0 */
    z[id] = x[id] * y[id];
}

int main()
{
    int a[6];
    int b[6];
    int c[6];
    int *d, *e, *f;
    int i;

    printf("\nEnter six elements of the first array:\n");
    for (i = 0; i < 6; i++)
    {
        scanf("%d", &a[i]);
    }

    printf("\nEnter six elements of the second array:\n");
    for (i = 0; i < 6; i++)
    {
        scanf("%d", &b[i]);
    }

    /* cudaMalloc() allocates memory from Global memory on GPU */
    cudaMalloc((void **)&d, 6 * sizeof(int));
    cudaMalloc((void **)&e, 6 * sizeof(int));
    cudaMalloc((void **)&f, 6 * sizeof(int));

    /* cudaMemcpy() copies the contents from the destination to the source. Here the destination is GPU (d, e)
    and the source is CPU (a, b) */
    cudaMemcpy(d, a, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, b, 6 * sizeof(int), cudaMemcpyHostToDevice);

    /* Call the kernel. Here 6 is the number of blocks, 1 is the number of threads per block, and d, e, f are the arguments */
    arrmul<<<6, 1>>>(d, e, f);

    /* Copy the result from GPU (Device) to CPU (Host) */
    cudaMemcpy(c, f, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nMultiplication of two arrays:\n");
    for (i = 0; i < 6; i++)
    {
        printf("%d\t", c[i]);
    }

    /* Free the memory allocated to pointers d, e, f */
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}
