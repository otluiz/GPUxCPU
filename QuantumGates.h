#ifndef QUANTUMGATES_H
#define QUANTUMGATES_H

#include <Eigen/Dense>

using namespace Eigen;

namespace QuantumGates {

  Matrix2cd hadamardGate(); //1

  Matrix2cd pauliXGate(); //2

  Matrix2cd pauliYGate(); //3

  Matrix2cd pauliZGate(); //4

  Matrix2cd sGate(); //5
  
  Matrix2cd sDaggerGate(); //6
  
  Matrix4cd toffoliGate(); //7
  
  Matrix4cd freadkinGate(); //8
  
  Matrix4cd qSwapGate();

  Matrix4cd cnotGate();

  
// A biblioteca Eigen deve estar no include dos arquivos

} // namespace QuantumGates

#endif // QUANTUMGATES_H
