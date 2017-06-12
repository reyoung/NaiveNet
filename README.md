# Naive Net

Experimental neural network implementation for investigating computation graph.

The lines of code should be less than 2000 or 3000, and should complete the following works:

1. [Done] Register computation graph meta information.
2. [Done] User can configure a computation graph.
3. [Done] forward a fast forward network.
4. [Done] backward a fast forward network.
5. [Done] Abstract Graph Compiler concept.
5. [Done] Optimization.
5. [Done] Workspace.
5. [Doing] Support sparse data type for NLP.
6. [TODO] Recurrent Neural Network.
7. [TODO] Dynamic Network.
8. [TODO] MultiThread Engine.

## Build & RUN

```cpp
git clone --recursive https://github.com/reyoung/NaiveNet.git
cd NaiveNet
mkdir build
cd build
cmake ..
make
cd ..
./build/NaiveNet
```
