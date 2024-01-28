#include <stdio.h>

// Kernel CUDA para somar dois vetores
__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void) {
    int N = 1<<20; // 1 milhão de elementos
    float *x, *y, *d_x, *d_y;

    // Aloca memória no host
    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    // Inicializa os vetores no host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Aloca memória na GPU
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    // Copia os vetores para a GPU
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Executa o kernel add() na GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, d_x, d_y);

    // Copia os resultados de volta para o host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("y[0] = %f\n", y[0]);

    // Libera a memória
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    
    return 0;
}
