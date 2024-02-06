#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "QuantumGates.h"

using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::MatrixXcd;
using namespace QuantumGates;

Eigen::MatrixXcd kroneckerProduct(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {
    Eigen::MatrixXcd C(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }
    return C;
}


int main() {
    int numQubits = 4;
    int N = 1 << numQubits; // 2^4 = 16
    MatrixXcd qubitState = MatrixXcd::Zero(N, 1);
    qubitState(12, 0) = 1; // Estado inicial |1100>

    // Cria a porta Hadamard para 4 qubits
    Matrix2cd h = hadamardGate();
 	MatrixXcd hadamard4 = kroneckerProduct(kroneckerProduct(h, h), kroneckerProduct(h, h));

	// Aplica a porta Hadamard ao estado do qubit
    qubitState = hadamard4 * qubitState;

	
	Eigen::MatrixXcd identity2 = Eigen::MatrixXcd::Identity(2, 2); // Matriz de Identidade 2x2
    Eigen::MatrixXcd identity4 = kroneckerProduct(identity2, identity2); // Expande para 4x4

	// Aplica a porta CNOT
    Matrix4cd cnot = cnotGate();
 	MatrixXcd cnot4 = kroneckerProduct(identity4, cnot);

	// Aplica a porta CNot ao estado do qubit
    qubitState = cnot4 * qubitState;

	 // Imprime o estado do qubit
    std::cout << "Estado do qubit:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << qubitState(i, 0) << " ";
    }
    std::cout << std::endl;

	// Após aplicar as portas...
	std::cout << "Probabilidades dos estados após Hadamard e CNOT:\n";
	for (int i = 0; i < 15; ++i) {
	  double prob = std::norm(qubitState(i, 0)); // Módulo ao quadrado da amplitude
	  std::cout << "P(|" << i << ">) = " << prob << "\n";
	}

	
    return 0;
}

/*
  g++ qubit5_simulation.cpp ../QuantumGates.cpp -o qubit5_HxC -I..

 */
