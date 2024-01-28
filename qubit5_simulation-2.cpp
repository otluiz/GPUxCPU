#include <iostream>
#include <Eigen/Dense>
#include "QuantumGates.h" // arquivo no mesmo diretório

using QuantumGates::hadamardGate; // usa namespace QuantumGates
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::MatrixXcd;

// Este método aumenta a dimensionalidade de uma matriz 2x2 para uma matriz
// Queremos 5 qubits ou seja uma matriz // 2^5 = 32 (32 x 32) mantendo suas propriedades
Eigen::Matrix4cd kroneckerProduct2x2(const Eigen::Matrix2cd& A, const Eigen::Matrix2cd& B) {
         Eigen::Matrix4cd C;
              for (int i = 0; i < 2; ++i) {
                  for (int j = 0; j < 2; ++j) {
                      C.block<2,2>(2*i, 2*j) = A(i,j) * B;
                  }
              }
              return C;
    }

MatrixXcd createHadamardGateFor5Qubits() {
    Matrix2cd h = hadamardGate();
    MatrixXcd result = h;

    for (int i = 1; i < 5; ++i) {
        result = kroneckerProduct2x2(result, h);
    }

    return result;
}

int main() {
    int numQubits = 5;
    int N = 1 << numQubits; // 2^5 = 32

    MatrixXcd qubitState = MatrixXcd::Zero(N, 1); // Estado inicial |00000>
    qubitState(0, 0) = 1;

    // Cria a porta Hadamard para 5 qubits
    MatrixXcd hadamard5Qubits = createHadamardGateFor5Qubits();

    // Aplica a porta Hadamard ao estado do qubit
    qubitState = hadamard5Qubits * qubitState;

    // Implementar e aplicar a porta CNOT aqui

    // Imprime o estado do qubit (opcional e somente para verificação)
    std::cout << "Estado do qubit:\n" << qubitState << std::endl;

    return 0;
}
