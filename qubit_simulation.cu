#include <cuda_runtime.h>
#include <iostream>

__global__ void applyGateKernel(float *qubit, const float *gate, int size) {
    int i = threadIdx.x;
    float temp = 0.0f;
    for (int j = 0; j < size; j++) {
        temp += gate[i * size + j] * qubit[j];
    }
    qubit[i] = temp;
}

int main() {
    int N = 2; // Tamanho do vetor para um qubit

    // Aloca memória no host
    float *h_qubit = (float*)malloc(N * sizeof(float));
    float *h_gate = (float*)malloc(N * N * sizeof(float));

    // Inicializa o qubit no estado |0> e a porta Hadamard
    h_qubit[0] = 1.0f; h_qubit[1] = 0.0f;
    h_gate[0] = 1.0f / sqrt(2.0f); h_gate[1] = 1.0f / sqrt(2.0f);
    h_gate[2] = 1.0f / sqrt(2.0f); h_gate[3] = -1.0f / sqrt(2.0f);

    // Aloca memória na GPU
    float *d_qubit, *d_gate;
    cudaMalloc(&d_qubit, N * sizeof(float));
    cudaMalloc(&d_gate, N * N * sizeof(float));

    // Copia os vetores para a GPU
    cudaMemcpy(d_qubit, h_qubit, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Executa o kernel na GPU
    applyGateKernel<<<1, N>>>(d_qubit, d_gate, N);

    // Copia os resultados de volta para o host
    cudaMemcpy(h_qubit, d_qubit, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Imprime o estado do qubit
    std::cout << "Estado do qubit: [" << h_qubit[0] << ", " << h_qubit[1] << "]" << std::endl;

    // Libera a memória
    cudaFree(d_qubit);
    cudaFree(d_gate);
    free(h_qubit);
    free(h_gate);

    return 0;
}
