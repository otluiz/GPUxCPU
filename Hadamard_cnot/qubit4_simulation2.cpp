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
    
    // Estado inicial |1100>
    qubitState(12, 0) = 1; 

    // Aplica a porta Hadamard (opcional)
    // Descomente a linha abaixo para ver o efeito da Hadamard
    // qubitState = kroneckerProduct(kroneckerProduct(hadamardGate(), hadamardGate()), kroneckerProduct(hadamardGate(), hadamardGate())) * qubitState;

	Eigen::MatrixXcd identity2 = Eigen::MatrixXcd::Identity(2, 2); // Matriz de Identidade 2x2
	
    // Aplica a porta CNOT
    Matrix4cd cnot = cnotGate();
    MatrixXcd cnot4 = kroneckerProduct(kroneckerProduct(cnot, identity2), identity2);
    qubitState = cnot4 * qubitState;

    // Imprime o estado do qubit
    std::cout << "Estado do qubit após CNOT:\n";
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
