#include <iostream>
#include <vector>
#include <cmath>

// Representação de um qubit como um vetor
using Qubit = std::vector<float>;

// Função para aplicar uma porta quântica
void applyGate(Qubit& qubit, const std::vector<std::vector<float>>& gate) {
    Qubit result(2, 0.0f);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            result[i] += gate[i][j] * qubit[j];
        }
    }
    qubit = result;
}

// Porta Hadamard
const std::vector<std::vector<float>> HADAMARD = {
    {1 / sqrt(2), 1 / sqrt(2)},
    {1 / sqrt(2), -1 / sqrt(2)}
};

int main() {
    // Inicializa um qubit no estado |0>
    Qubit qubit = {1, 0};

    // Aplica a porta Hadamard
    applyGate(qubit, HADAMARD);

    // Imprime o estado do qubit
    std::cout << "Estado do qubit: [" << qubit[0] << ", " << qubit[1] << "]" << std::endl;

    return 0;
}
