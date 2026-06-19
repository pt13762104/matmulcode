# matmulcode
Sample matrix multiplication codes

## int8_96x96.cpp

This does int8xint8->int32 multiplication. Outputs to stderr the performance for matrix sizes of size $2^k*96$ with $0 \leq k \leq 7$. The program outputs two value each, one for the raw matmul step, and one includes the conversion step. Currently only supports matrix sizes of $2^k*96$ for now...

Requires Armv8 with SVE and I8MM (Cortex-X2/A710/A510 or later.)

