#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "QuantumGates.h" // arquivo no mesmo diretório

using QuantumGates::hadamardGate; // usa namespace QuantumGates
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::MatrixXcd;

// Este método aumenta a dimensionalidade de uma matriz 2x2 para uma matriz
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
    Matrix2cd h = hadamardGate(); // Supondo que esta função retorna a matriz Hadamard 2x2
    Matrix4cd hadamard2Qubits = kroneckerProduct2x2(h, h); // Porta Hadamard para 2 qubits

    MatrixXcd qubitState = MatrixXcd::Zero(4, 1); // Estado inicial |00> para 2 qubits
    qubitState(0, 0) = 1;

    qubitState = hadamard2Qubits * qubitState; // Aplica a porta Hadamard

    // Imprime o estado do qubit
    std::cout << "Estado do qubit:\n" << qubitState << std::endl;

    return 0;
}
