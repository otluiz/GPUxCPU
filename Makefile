CC = g++
CFLAGS = -std=c++11 -I/usr/include/eigen3/Eigen/  # Caminho para a biblioteca Eigen
LDFLAGS =

# Lista de arquivos objeto
OBJ = qubit5_simulation-2.o QuantumGates.o

# Nome do executável
TARGET = qubit5_simulation-2

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

qubit5_simulation-2.o: qubit5_simulation-2.cpp
	$(CC) $(CFLAGS) -c qubit5_simulation-2.cpp

QuantumGates.o: QuantumGates.cpp QuantumGates.h
	$(CC) $(CFLAGS) -c QuantumGates.cpp

clean:
	rm -f $(TARGET) $(OBJ)
