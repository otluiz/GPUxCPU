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
    int numQubits = 5;
    int N = 1 << numQubits; // 2^5 = 32
    MatrixXcd qubitState = MatrixXcd::Zero(N, 1);
    qubitState(0, 0) = 1; // Estado inicial |00000>

    // ... aplicação de portas quânticas ...

	// Cria a porta Hadamard para 5 qubits
    Matrix2cd h = hadamardGate();
    Matrix4cd hadamard5Qubits = kroneckerProduct2x2(h, h);
    
    // Aplica a porta Hadamard ao estado do qubit
    qubitState = hadamard5Qubits * qubitState;

    // Aplica a porta CNOT
    Matrix4cd cnot = cnotGate();
    qubitState = cnot * qubitState;

	

    // Imprime o estado do qubit
    std::cout << "Estado do qubit:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << qubitState(i, 0) << " ";
    }
    std::cout << std::endl;

	// Após aplicar as portas...
	std::cout << "Probabilidades dos estados após Hadamard e CNOT:\n";
	for (int i = 0; i < 4; ++i) {
	  double prob = std::norm(qubitState(i, 0)); // Módulo ao quadrado da amplitude
	  std::cout << "P(|" << i << ">) = " << prob << "\n";
	}

	
    return 0;
}
