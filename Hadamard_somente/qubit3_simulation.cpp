#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include "QuantumGates.h" // arquivo no mesmo diretório

using QuantumGates::hadamardGate; // usa namespace QuantumGates
using Eigen::Matrix2cd;
using Eigen::Matrix4cd;
using Eigen::MatrixXcd;

// Este método aumenta a dimensionalidade de uma matriz 2x2 para uma matriz
// Queremos 3 qubits ou seja uma matriz // 2^5 = 32 (32 x 32) mantendo suas propriedades
Eigen::MatrixXcd kroneckerProduct3x3(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {
    Eigen::MatrixXcd C(A.rows() * B.rows(), A.cols() * B.cols());

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            C.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }

    return C;
}


MatrixXcd createHadamardGateFor3Qubits() {
    Matrix2cd h = hadamardGate(); // Função que retorna a matriz Hadamard 2x2
    MatrixXcd result = h;

    for (int i = 1; i < 3; ++i) { // Repete 2 vezes mais para ter 3 qubits
        result = kroneckerProduct3x3(result, h);
    }

    return result;
}


int main() {
    MatrixXcd qubitState = MatrixXcd::Zero(8, 1); // Estado inicial |000> para 3 qubits
    qubitState(0, 0) = 1;

    MatrixXcd hadamard3Qubits = createHadamardGateFor3Qubits();
    qubitState = hadamard3Qubits * qubitState;

    std::cout << "Estado do qubit:\n" << qubitState << std::endl;
    return 0;
}
