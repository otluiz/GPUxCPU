#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "QuantumGates.h"

using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::MatrixXcd;
using namespace QuantumGates;

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


int main() {
    // Estado inicial |00>
    MatrixXcd qubitState = MatrixXcd::Zero(4, 1);
    qubitState(0, 0) = 1;

    // Cria a porta Hadamard para 2 qubits
    Matrix2cd h = hadamardGate();
    Matrix4cd hadamard2Qubits = kroneckerProduct2x2(h, h);
    
    // Aplica a porta Hadamard ao estado do qubit
    qubitState = hadamard2Qubits * qubitState;

    // Aplica a porta CNOT
    Matrix4cd cnot = cnotGate();
    qubitState = cnot * qubitState;

    // Imprime o estado do qubit
    std::cout << "Estado do qubit após Hadamard e CNOT:\n" << qubitState << std::endl;

    return 0;
}

/*
 * Diretiva de compilação para procurar o arquivo de cabeçalho  no diretório acima
 * g++ qubit2_simulation.cpp ../QuantumGates.cpp -o qubit2_HxC -I..
 */
 
