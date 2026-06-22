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

## int8_128x128.cpp

Does the same thing as int8_96x96 but with the kernel size being 128x128. An experiment to see if C could be outside L1 or not (A and B shouldn't), and it's a success.

### Performance
![](https://raw.githubusercontent.com/pt13762104/matmulcode/refs/heads/main/perf_128x128.png)

## int8_general.cpp
Replacement for int8_96x96 and int8_128x128. Now, the base size could be anything that's divisible by 64. This was inspired from q8_0_4x4 where it shows that fitting in L1 isn't the best strategy. 

This interleave 4 row-blocks (of size 2) to make sure the A- and B- tiles were always loaded sequentially. Every iteration of the j-loop, the program does 16 `smmla`s to do a full 8x8x8 matrix multiplication.

### Performance
![](https://raw.githubusercontent.com/pt13762104/matmulcode/refs/heads/main/perf_int8.png)

## q8_0_4x4.cpp
Re-implementation of the q8_0_4x4 format of llama.cpp. Performance is equal to llama.cpp at size 4096 and above, but is better at lower matrix sizes. There's no repacking code at the moment.

### Performance
![](https://raw.githubusercontent.com/pt13762104/matmulcode/refs/heads/main/perf_q8_0_4_4.png)

## fp32_32x32.cpp

This does fp32xfp32->fp32 multiplication. Outputs to stdout the performance for matrix sizes $4096\times4096$, including the conversion step. Currently only supports matrix sizes of $2^k\times32$ for now...

Requires x86 with AVX2 (Haswell/Zen or later.)

The kernel is 32x32 with 2x4 tiling (broadcasting two values of A and load 4 B vectors. C has 2x4=8 regs).

The 32x32 blocks of A, B, C were kept in original order (row-major assumed).

## Conversion step

The conversion step is as follows:

Assume a matrix $M$ of size $N$. Recursively convert the 4 quadrants of size $N/2\times N/2$ and write them in order to the converted array. When $N$ is the tile size above, the format is defined as above.
