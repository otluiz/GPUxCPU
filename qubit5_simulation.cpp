#include <iostream>
#include <vector>
#include <cmath>

using Qubit = std::vector<float>;

void applyGate(Qubit& qubit, const std::vector<std::vector<float>>& gate) {
    int size = qubit.size();
    Qubit result(size, 0.0f);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            result[i] += gate[i][j] * qubit[j];
        }
    }
    qubit = result;
}

// Esta função precisa ser implementada para criar a matriz da porta para 5 qubits
std::vector<std::vector<float>> createGateMatrixFor5Qubits() {
    // Implementação depende da porta específica que você quer usar
    // Exemplo: retornar uma matriz identidade de 32x32
    int size = 32;
    std::vector<std::vector<float>> gate(size, std::vector<float>(size, 0.0f));
    for (int i = 0; i < size; ++i) {
        gate[i][i] = 1.0f;
    }
    return gate;
}

int main() {
    int numQubits = 5;
    int stateSize = pow(2, numQubits);
    Qubit qubit(stateSize, 0.0f); // Estado inicial |00000>

    // Cria a matriz da porta para 5 qubits
    std::vector<std::vector<float>> gateMatrix = createGateMatrixFor5Qubits();

    // Aplica a porta ao qubit
    applyGate(qubit, gateMatrix);

    // Imprime o estado do qubit
    std::cout << "Estado do qubit: [";
    for (float q : qubit) {
        std::cout << q << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}
