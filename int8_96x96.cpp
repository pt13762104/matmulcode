#include <arm_sve.h>
#include <bits/stdc++.h>

#include <cassert>
using namespace std;
#ifndef yoshi_likes_e4
#define endl '\n'
#endif
#define problem ""
#define multitest 0
#define debug(x) cerr << #x << " = " << x << endl;
void init()
{
}
// 96 x 96 I8 x I8 -> I32
void matmulX(int8_t *A, int8_t *B, int32_t *C)
{
    A = (int8_t *)__builtin_assume_aligned(A, 16);
    B = (int8_t *)__builtin_assume_aligned(B, 16);
    C = (int32_t *)__builtin_assume_aligned(C, 16);
    for (int i = 0; i < 96; i += 8)
        for (int k = 0; k < 96; k += 8)
        {
            auto Cr0 = svld1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2) * 4);
            auto Cr1 = svld1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 1) * 4);
            auto Cr2 = svld1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 2) * 4);
            auto Cr3 = svld1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 3) * 4);
            auto Cr4 = svld1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2) * 4);
            auto Cr5 = svld1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 1) * 4);
            auto Cr6 = svld1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 2) * 4);
            auto Cr7 = svld1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 3) * 4);
            auto Cr8 = svld1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2) * 4);
            auto Cr9 = svld1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 1) * 4);
            auto Cr10 = svld1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 2) * 4);
            auto Cr11 = svld1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 3) * 4);
            auto Cr12 = svld1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2) * 4);
            auto Cr13 = svld1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 1) * 4);
            auto Cr14 = svld1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 2) * 4);
            auto Cr15 = svld1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 3) * 4);
            for (int j = 0; j < 96; j += 8)
            {
                auto ai1 = svld1_s8(svptrue_b8(), A + (j / 8) * 768 + (i / 8) * 64);
                auto ai2 = svld1_s8(svptrue_b8(), A + (j / 8) * 768 + (i / 8) * 64 + 16);
                auto ai3 = svld1_s8(svptrue_b8(), A + (j / 8) * 768 + (i / 8) * 64 + 32);
                auto ai4 = svld1_s8(svptrue_b8(), A + (j / 8) * 768 + (i / 8) * 64 + 48);
                auto bi1 = svld1_s8(svptrue_b8(), B + (j / 8) * 768 + (k / 8) * 64);
                auto bi2 = svld1_s8(svptrue_b8(), B + (j / 8) * 768 + (k / 8) * 64 + 16);
                auto bi3 = svld1_s8(svptrue_b8(), B + (j / 8) * 768 + (k / 8) * 64 + 32);
                auto bi4 = svld1_s8(svptrue_b8(), B + (j / 8) * 768 + (k / 8) * 64 + 48);
                Cr0 = svmmla_s32(Cr0, ai1, bi1);
                Cr1 = svmmla_s32(Cr1, ai1, bi2);
                Cr2 = svmmla_s32(Cr2, ai1, bi3);
                Cr3 = svmmla_s32(Cr3, ai1, bi4);
                Cr4 = svmmla_s32(Cr4, ai2, bi1);
                Cr5 = svmmla_s32(Cr5, ai2, bi2);
                Cr6 = svmmla_s32(Cr6, ai2, bi3);
                Cr7 = svmmla_s32(Cr7, ai2, bi4);
                Cr8 = svmmla_s32(Cr8, ai3, bi1);
                Cr9 = svmmla_s32(Cr9, ai3, bi2);
                Cr10 = svmmla_s32(Cr10, ai3, bi3);
                Cr11 = svmmla_s32(Cr11, ai3, bi4);
                Cr12 = svmmla_s32(Cr12, ai4, bi1);
                Cr13 = svmmla_s32(Cr13, ai4, bi2);
                Cr14 = svmmla_s32(Cr14, ai4, bi3);
                Cr15 = svmmla_s32(Cr15, ai4, bi4);
            }
            svst1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2) * 4, Cr0);
            svst1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 1) * 4, Cr1);
            svst1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 2) * 4, Cr2);
            svst1_s32(svptrue_b32(), C + (i / 2) * 192 + (k / 2 + 3) * 4, Cr3);
            svst1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2) * 4, Cr4);
            svst1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 1) * 4, Cr5);
            svst1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 2) * 4, Cr6);
            svst1_s32(svptrue_b32(), C + (i / 2 + 1) * 192 + (k / 2 + 3) * 4, Cr7);
            svst1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2) * 4, Cr8);
            svst1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 1) * 4, Cr9);
            svst1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 2) * 4, Cr10);
            svst1_s32(svptrue_b32(), C + (i / 2 + 2) * 192 + (k / 2 + 3) * 4, Cr11);
            svst1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2) * 4, Cr12);
            svst1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 1) * 4, Cr13);
            svst1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 2) * 4, Cr14);
            svst1_s32(svptrue_b32(), C + (i / 2 + 3) * 192 + (k / 2 + 3) * 4, Cr15);
        }
}
constexpr int NX = 12288;
alignas(32) int8_t tmp1[NX * NX];
alignas(32) int8_t tmp2[NX * NX];
alignas(32) int32_t tmp3[NX * NX];
alignas(32) int8_t A_test[NX * NX];
alignas(32) int8_t B_test[NX * NX];
alignas(32) int32_t C_test[NX * NX];
template <int S> void recurse(int A, int B, int C)
{
    if (S == 96)
        matmulX(tmp1 + A, tmp2 + B, tmp3 + C);
    else
    {
        // C[0,0] += A[0,0] * B[0,0]
        recurse<S / 2>(A, B, C);
        // C[0,1] += A[0,0] * B[0,1]
        recurse<S / 2>(A, B + S * S / 4, C + S * S / 4);
        // C[0,0] += A[0,1] * B[1,0]
        recurse<S / 2>(A + S * S / 4, B + 2 * S * S / 4, C);
        // C[0,1] += A[0,1] * B[1,1]
        recurse<S / 2>(A + S * S / 4, B + 3 * S * S / 4, C + S * S / 4);
        // C[1,0] += A[1,0] * B[0,0]
        recurse<S / 2>(A + 2 * S * S / 4, B, C + 2 * S * S / 4);
        // C[1,1] += A[1,0] * B[0,1]
        recurse<S / 2>(A + 2 * S * S / 4, B + S * S / 4, C + 3 * S * S / 4);
        // C[1,0] += A[1,1] * B[1,0]
        recurse<S / 2>(A + 3 * S * S / 4, B + 2 * S * S / 4, C + 2 * S * S / 4);
        // C[1,1] += A[1,1] * B[1,1]
        recurse<S / 2>(A + 3 * S * S / 4, B + 3 * S * S / 4, C + 3 * S * S / 4);
    }
}
// repack src from row-major to tree-form
template <int N, int NX> void repackA(int8_t *__restrict__ dest, int8_t *__restrict__ src, int tl, int tr)
{
    src = (int8_t *)__builtin_assume_aligned(src, 16);
    dest = (int8_t *)__builtin_assume_aligned(dest, 16);
    if (N == 96)
    {
        // copy, 2x8 row-wise transposed
        for (int i = 0; i < 96; i += 2)
            for (int j = 0; j < 96; j += 8)
                for (int k = 0; k < 2; k++)
                    for (int t = 0; t < 8; t++)
                        dest[(j / 8) * 768 + (i / 2) * 16 + k * 8 + t] = src[(tl + i + k) * NX + tr + j + t];
    }
    else
    {
        repackA<N / 2, NX>(dest, src, tl, tr);
        repackA<N / 2, NX>(dest + N * N / 4, src, tl, tr + N / 2);
        repackA<N / 2, NX>(dest + 2 * N * N / 4, src, tl + N / 2, tr);
        repackA<N / 2, NX>(dest + 3 * N * N / 4, src, tl + N / 2, tr + N / 2);
    }
}
template <int N, int NX> void repackB(int8_t *__restrict__ dest, int8_t *__restrict__ src, int tl, int tr)
{
    src = (int8_t *)__builtin_assume_aligned(src, 16);
    dest = (int8_t *)__builtin_assume_aligned(dest, 16);
    if (N == 96)
    {
        // copy, 8x2 column-wise
        for (int i = 0; i < 96; i += 8)
            for (int j = 0; j < 96; j += 2)
                for (int k = 0; k < 8; k++)
                    for (int t = 0; t < 2; t++)
                        dest[(i / 8) * 768 + (j / 2) * 16 + t * 8 + k] = src[(tl + i + k) * NX + tr + j + t];
    }
    else
    {
        repackB<N / 2, NX>(dest, src, tl, tr);
        repackB<N / 2, NX>(dest + N * N / 4, src, tl, tr + N / 2);
        repackB<N / 2, NX>(dest + 2 * N * N / 4, src, tl + N / 2, tr);
        repackB<N / 2, NX>(dest + 3 * N * N / 4, src, tl + N / 2, tr + N / 2);
    }
}
template <int N, int NX> void repackC(int *__restrict__ dest, int *__restrict__ src, int tl, int tr)
{
    src = (int *)__builtin_assume_aligned(src, 16);
    dest = (int *)__builtin_assume_aligned(dest, 16);
    if (N == 96)
    {
        // copy, 2x2 row-wise
        for (int i = 0; i < 96; i += 2)
            for (int j = 0; j < 96; j += 2)
                for (int k = 0; k < 2; k++)
                    for (int t = 0; t < 2; t++)
                        dest[(tl + i + k) * NX + tr + j + t] = src[(i / 2) * 192 + (j / 2) * 4 + k * 2 + t];
    }
    else
    {
        repackC<N / 2, NX>(dest, src, tl, tr);
        repackC<N / 2, NX>(dest, src + N * N / 4, tl, tr + N / 2);
        repackC<N / 2, NX>(dest, src + 2 * N * N / 4, tl + N / 2, tr);
        repackC<N / 2, NX>(dest, src + 3 * N * N / 4, tl + N / 2, tr + N / 2);
    }
}
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
int D[96 * 96];
template <int NX> void matmultest()
{
    auto aa = clock();
    repackA<NX, NX>(tmp1, A_test, 0, 0);
    repackB<NX, NX>(tmp2, B_test, 0, 0);
    auto a = clock();
    recurse<NX>(0, 0, 0);
    auto b = clock();
    repackC<NX, NX>(C_test, tmp3, 0, 0);
    auto bb = clock();
    cerr << NX << "," << 2.0 * NX * NX * NX / 1000 / (b - a) << ",";
    cerr << 2.0 * NX * NX * NX / 1000 / (bb - aa) << endl;
}
template <int NX> void verif()
{
    for (int i = 0; i < 96 * 96; i++)
        A_test[i] = rng();
    for (int i = 0; i < 96 * 96; i++)
        B_test[i] = rng();
    for (int i = 0; i < 96; i++)
        for (int j = 0; j < 96; j++)
            for (int k = 0; k < 96; k++)
                D[i * 96 + k] += (short)(A_test[i * 96 + j]) * B_test[j * 96 + k];
    auto aa = clock();
    repackA<NX, NX>(tmp1, A_test, 0, 0);
    repackB<NX, NX>(tmp2, B_test, 0, 0);
    auto a = clock();
    recurse<NX>(0, 0, 0);
    auto b = clock();
    repackC<NX, NX>(C_test, tmp3, 0, 0);
    auto bb = clock();
    for (int i = 0; i < 96 * 96; i++)
        assert(C_test[i] == D[i]);
    cerr << NX << "," << 2.0 * NX * NX * NX / 1000 / (b - a) << ",";
    cerr << 2.0 * NX * NX * NX / 1000 / (bb - aa) << endl;
}
void Yoshi()
{
    // This kernel is for 128-bit SVE only
    assert(svcntb() == 16);
    // 96x96 MATMUL
    matmultest<12288>();
    matmultest<12288>();
    matmultest<12288>();
    matmultest<12288>();
    matmultest<12288>();
    verif<96>();
    matmultest<192>();
    matmultest<384>();
    matmultest<768>();
    matmultest<1536>();
    matmultest<3072>();
    matmultest<6144>();
    matmultest<12288>();
}
signed main()
{
#ifndef yoshi_likes_e4
    ios::sync_with_stdio(0);
    if (fopen(problem ".inp", "r"))
    {
        freopen(problem ".inp", "r", stdin);
        freopen(problem ".out", "w", stdout);
    }
#endif
    init();
    int t = 1;
#if multitest
    cin >> t;
#endif
    while (t--)
        Yoshi();
}
