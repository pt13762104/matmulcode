#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std;
alignas(32) double B[256 * 256];
alignas(32) double C[256 * 256];
// 3x16 tiling (thanks libxsmm)
void AVX2_EMM_TILE(const double *__restrict__ A, double *__restrict__ C, int IT, int JT, int KT)
{
    // 3 x 16 x 64 matmul
    __m256d Cr0 = _mm256_load_pd(C + (IT * 3) * 256 + JT * 16), Cr1 = _mm256_load_pd(C + (IT * 3) * 256 + JT * 16 + 4),
            Cr2 = _mm256_load_pd(C + (IT * 3) * 256 + JT * 16 + 8),
            Cr3 = _mm256_load_pd(C + (IT * 3) * 256 + JT * 16 + 12),
            Cr4 = _mm256_load_pd(C + (IT * 3 + 1) * 256 + JT * 16 + 0),
            Cr5 = _mm256_load_pd(C + (IT * 3 + 1) * 256 + JT * 16 + 4),
            Cr6 = _mm256_load_pd(C + (IT * 3 + 1) * 256 + JT * 16 + 8),
            Cr7 = _mm256_load_pd(C + (IT * 3 + 1) * 256 + JT * 16 + 12),
            Cr8 = _mm256_load_pd(C + (IT * 3 + 2) * 256 + JT * 16 + 0),
            Cr9 = _mm256_load_pd(C + (IT * 3 + 2) * 256 + JT * 16 + 4),
            Cr10 = _mm256_load_pd(C + (IT * 3 + 2) * 256 + JT * 16 + 8),
            Cr11 = _mm256_load_pd(C + (IT * 3 + 2) * 256 + JT * 16 + 12);
#pragma GCC unroll 0
    for (int k = 0; k < 64; k++)
    {
        auto ai = _mm256_broadcast_sd(A + (IT * 3) * 256 + k + KT * 64);
        auto ai2 = _mm256_broadcast_sd(A + (IT * 3 + 1) * 256 + k + KT * 64);
        auto ai3 = _mm256_broadcast_sd(A + (IT * 3 + 2) * 256 + k + KT * 64);
        auto bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16);
        Cr0 = _mm256_fmadd_pd(ai, bi, Cr0);
        Cr4 = _mm256_fmadd_pd(ai2, bi, Cr4);
        Cr8 = _mm256_fmadd_pd(ai3, bi, Cr8);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 4);
        Cr1 = _mm256_fmadd_pd(ai, bi, Cr1);
        Cr5 = _mm256_fmadd_pd(ai2, bi, Cr5);
        Cr9 = _mm256_fmadd_pd(ai3, bi, Cr9);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 8);
        Cr2 = _mm256_fmadd_pd(ai, bi, Cr2);
        Cr6 = _mm256_fmadd_pd(ai2, bi, Cr6);
        Cr10 = _mm256_fmadd_pd(ai3, bi, Cr10);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 12);
        Cr3 = _mm256_fmadd_pd(ai, bi, Cr3);
        Cr7 = _mm256_fmadd_pd(ai2, bi, Cr7);
        Cr11 = _mm256_fmadd_pd(ai3, bi, Cr11);
    }
    *(__m256d *)(C + (IT * 3) * 256 + JT * 16) = Cr0;
    *(__m256d *)(C + (IT * 3) * 256 + JT * 16 + 4) = Cr1;
    *(__m256d *)(C + (IT * 3) * 256 + JT * 16 + 8) = Cr2;
    *(__m256d *)(C + (IT * 3) * 256 + JT * 16 + 12) = Cr3;
    *(__m256d *)(C + (IT * 3 + 1) * 256 + JT * 16) = Cr4;
    *(__m256d *)(C + (IT * 3 + 1) * 256 + JT * 16 + 4) = Cr5;
    *(__m256d *)(C + (IT * 3 + 1) * 256 + JT * 16 + 8) = Cr6;
    *(__m256d *)(C + (IT * 3 + 1) * 256 + JT * 16 + 12) = Cr7;
    *(__m256d *)(C + (IT * 3 + 2) * 256 + JT * 16) = Cr8;
    *(__m256d *)(C + (IT * 3 + 2) * 256 + JT * 16 + 4) = Cr9;
    *(__m256d *)(C + (IT * 3 + 2) * 256 + JT * 16 + 8) = Cr10;
    *(__m256d *)(C + (IT * 3 + 2) * 256 + JT * 16 + 12) = Cr11;
}
void AVX2_EMM_TILE_2(const double *__restrict__ A, double *__restrict__ C, int IT, int JT, int KT)
{
    // 2 x 16 x 64 matmul
    __m256d Cr0 = _mm256_load_pd(C + (IT * 2) * 256 + JT * 16), Cr1 = _mm256_load_pd(C + (IT * 2) * 256 + JT * 16 + 4),
            Cr2 = _mm256_load_pd(C + (IT * 2) * 256 + JT * 16 + 8),
            Cr3 = _mm256_load_pd(C + (IT * 2) * 256 + JT * 16 + 12),
            Cr4 = _mm256_load_pd(C + (IT * 2 + 1) * 256 + JT * 16 + 0),
            Cr5 = _mm256_load_pd(C + (IT * 2 + 1) * 256 + JT * 16 + 4),
            Cr6 = _mm256_load_pd(C + (IT * 2 + 1) * 256 + JT * 16 + 8),
            Cr7 = _mm256_load_pd(C + (IT * 2 + 1) * 256 + JT * 16 + 12);
#pragma GCC unroll 0
    for (int k = 0; k < 64; k++)
    {
        auto ai = _mm256_broadcast_sd(A + (IT * 2) * 256 + k + KT * 64);
        auto ai2 = _mm256_broadcast_sd(A + (IT * 2 + 1) * 256 + k + KT * 64);
        auto bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16);
        Cr0 = _mm256_fmadd_pd(ai, bi, Cr0);
        Cr4 = _mm256_fmadd_pd(ai2, bi, Cr4);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 4);
        Cr1 = _mm256_fmadd_pd(ai, bi, Cr1);
        Cr5 = _mm256_fmadd_pd(ai2, bi, Cr5);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 8);
        Cr2 = _mm256_fmadd_pd(ai, bi, Cr2);
        Cr6 = _mm256_fmadd_pd(ai2, bi, Cr6);
        bi = _mm256_load_pd(B + (JT * 16) * 256 + (k + KT * 64) * 16 + 12);
        Cr3 = _mm256_fmadd_pd(ai, bi, Cr3);
        Cr7 = _mm256_fmadd_pd(ai2, bi, Cr7);
    }
    *(__m256d *)(C + (IT * 2) * 256 + JT * 16) = Cr0;
    *(__m256d *)(C + (IT * 2) * 256 + JT * 16 + 4) = Cr1;
    *(__m256d *)(C + (IT * 2) * 256 + JT * 16 + 8) = Cr2;
    *(__m256d *)(C + (IT * 2) * 256 + JT * 16 + 12) = Cr3;
    *(__m256d *)(C + (IT * 2 + 1) * 256 + JT * 16) = Cr4;
    *(__m256d *)(C + (IT * 2 + 1) * 256 + JT * 16 + 4) = Cr5;
    *(__m256d *)(C + (IT * 2 + 1) * 256 + JT * 16 + 8) = Cr6;
    *(__m256d *)(C + (IT * 2 + 1) * 256 + JT * 16 + 12) = Cr7;
}
void thetruetest(const double *__restrict__ A, double *__restrict__ C)
{
    for (int ITB = 0; ITB < 84; ITB += 21)
        for (int JTB = 0; JTB < 16; JTB += 2)
            for (int KT = 0; KT < 4; KT++)
                for (int IT = ITB; IT < ITB + 21; IT++)
                    for (int JT = JTB; JT < JTB + 2; JT++)
                        AVX2_EMM_TILE(A, C, IT, JT, KT);
    for (int JTB = 0; JTB < 16; JTB += 4)
        for (int KT = 0; KT < 4; KT++)
            for (int IT = 126; IT < 128; IT++)
                for (int JT = JTB; JT < JTB + 4; JT++)
                    AVX2_EMM_TILE_2(A, C, IT, JT, KT);
}
double __attribute__((optimize("-ffast-math"))) multiply_matrices(double (*__restrict__ a)[256], double (*__restrict__ b)[256])
{
    for (auto &i : C)
        i = 0;
    for (int i = 0; i < 256; i++)
        for (int j = 0; j < 256; j += 16)
            for (int k = 0; k < 16; k++)
                B[j / 16 * 4096 + i * 16 + k] = b[i][j + k];
    thetruetest((double *)(a), C);
    double sum = 0;
    for (auto &i : C)
        sum += i * i;
    return sum;
}