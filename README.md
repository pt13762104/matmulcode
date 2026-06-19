# matmulcode
Sample matrix multiplication codes

## int8_96x96.cpp

This does int8xint8->int32 multiplication. Outputs to stderr the performance for matrix sizes $2^k\times96$ with $0 \leq k \leq 7$. The program outputs two value each, one for the raw matmul step, and one includes the conversion step. Currently only supports matrix sizes of $2^k\times96$ for now...

Requires Armv8 with SVE and I8MM (Cortex-X2/A710/A510 or later.)

The kernel is 96x96 with 8x8 tiling (16 regs of size 2x2, and 4+4 regs (as matrices of size 2x8 and 8x2) from A and B is loaded for each iteration of the j-loop).

Each 96x96 block of A is represented as a 12x48 (transposed) matrix, each element is an 2x8 matrix from the original A.

Each 96x96 block of B is represented as a 48x12 matrix, each element is an 2x8 (transposed) matrix from the original B.

Each 96x96 block of C is represented as a 48x48 matrix, each element is an 2x2 matrix.

### Performance
![](https://raw.githubusercontent.com/pt13762104/matmulcode/refs/heads/main/perf.png)

## fp32_32x32.cpp

This does fp32xfp32->fp32 multiplication. Outputs to stdout the performance for matrix sizes $4096\times4096$, including the conversion step. Currently only supports matrix sizes of $2^k\times32$ for now...

Requires x86 with AVX2 (Haswell/Zen or later.)

The kernel is 32x32 with 2x4 tiling (broadcasting two values of A and load 4 B vectors. C has 2x4=8 regs).

The 32x32 blocks of A, B, C were kept in original order (row-major assumed).

## Conversion step

The conversion step is as follows:

Assume a matrix $M$ of size $N$. Recursively convert the 4 quadrants of size $N/2\times N/2$ and write them in order to the converted array. When $N$ is the tile size above, the format is defined as above.
