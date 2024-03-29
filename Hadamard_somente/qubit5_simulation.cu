//;;Complexidade da Matriz da Porta: Definir operações quânticas
//;;para um sistema de 5 qubits pode ser bastante complexo.
//;;Aqui, usei uma matriz de identidade como um exemplo simples.

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void applyGateKernel(float *qubit, const float *gate, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        float temp = 0.0f;
        for (int j = 0; j < size; j++) {
            temp += gate[i * size + j] * qubit[j];
        }
        qubit[i] = temp;
    }
}

int main() {
    int numQubits = 5;
    int N = 1 << numQubits; // 2^5 para 5 qubits

    // Aloca memória no host
    float *h_qubit = (float*)malloc(N * sizeof(float));
    float *h_gate = (float*)malloc(N * N * sizeof(float));

    // Inicializa o qubit no estado |00000>
    for (int i = 0; i < N; i++) {
        h_qubit[i] = (i == 0) ? 1.0f : 0.0f;
    }
    // Inicializa a porta (exemplo: matriz identidade)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_gate[i * N + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Aloca memória na GPU
    float *d_qubit, *d_gate;
    cudaMalloc(&d_qubit, N * sizeof(float));
    cudaMalloc(&d_gate, N * N * sizeof(float));

    // Copia os vetores para a GPU
    cudaMemcpy(d_qubit, h_qubit, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define o número de blocos e threads
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Executa o kernel na GPU
    applyGateKernel<<<numBlocks, blockSize>>>(d_qubit, d_gate, N);

    // Copia os resultados de volta para o host
    cudaMemcpy(h_qubit, d_qubit, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Imprime o estado do qubit
    std::cout << "Estado do qubit: [";
    for (int i = 0; i < N; i++) {
        std::cout << h_qubit[i] << " ";
    }
    std::cout << "]" << std::endl;

    // Libera a memória
    cudaFree(d_qubit);
    cudaFree(d_gate);
    free(h_qubit);
    free(h_gate);

    return 0;
}
