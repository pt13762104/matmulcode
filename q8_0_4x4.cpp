#include <arm_neon.h>
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
// 128 x 128 Q8_0 x Q8_0 -> I32
// A has dimensions 128 x 128
// B has dimensions 128 x 128
// sA has dimensions 128 x 4 -> 32x16
// sB has dimensions 128 x 4 -> 32x16
// C has dimensions 128 x 128
template <int N> void matmulX(int8_t *A, int8_t *B, float16_t *sA, float16_t *sB, float *C)
{
    A = (int8_t *)__builtin_assume_aligned(A, 16);
    B = (int8_t *)__builtin_assume_aligned(B, 16);
    C = (float *)__builtin_assume_aligned(C, 16);
    sA = (float16_t *)__builtin_assume_aligned(sA, 16);
    sB = (float16_t *)__builtin_assume_aligned(sB, 16);
    for (int i = 0; i < N; i += 4)
        for (int k = 0; k < N; k += 4)
        {
            // C^T? bandwidth demands might be high. C is in L2 though.
            auto Cr0 = svld1_f32(svptrue_b32(), C + i * N + k);
            auto Cr1 = svld1_f32(svptrue_b32(), C + (i + 1) * N + k);
            auto Cr2 = svld1_f32(svptrue_b32(), C + (i + 2) * N + k);
            auto Cr3 = svld1_f32(svptrue_b32(), C + (i + 3) * N + k);
            // loadmaxx A, B, and scales
            // 32 is, in fact, QK8_0
            for (int jj = 0; jj < N; jj += 64)
            {
                // load scales assume permuted in [0, 4, 1, 5, 2, 6, 3, 7]
                auto scalesA = svld1_f16(svptrue_b16(), sA + (i / 4) * N / 8 + (jj / 64) * 8);
                auto scalesB = svld1_f16(svptrue_b16(), sA + (k / 4) * N / 8 + (jj / 64) * 8);
                for (int j = jj; j < jj + 64; j += 32)
                {
                    // Ct = 0
                    auto Ct0 = svdup_n_s32(0);
                    auto Ct1 = svdup_n_s32(0);
                    auto Ct2 = svdup_n_s32(0);
                    auto Ct3 = svdup_n_s32(0);
                    // loadmaxx A, B
                    auto ai0 = svld1_s8(svptrue_b8(), A + i * N + j * 4);
                    auto ai1 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 16);
                    auto ai2 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 32);
                    auto ai3 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 48);
                    auto ai4 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 64);
                    auto ai5 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 80);
                    auto ai6 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 96);
                    auto ai7 = svld1_s8(svptrue_b8(), A + i * N + j * 4 + 112);
                    auto bi0 = svld1_s8(svptrue_b8(), B + k * N + j * 4);
                    auto bi1 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 16);
                    auto bi2 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 32);
                    auto bi3 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 48);
                    auto bi4 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 64);
                    auto bi5 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 80);
                    auto bi6 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 96);
                    auto bi7 = svld1_s8(svptrue_b8(), B + k * N + j * 4 + 112);
                    Ct0 = svdot_s32(Ct0, ai0, bi0);
                    Ct0 = svdot_s32(Ct0, ai1, bi1);
                    Ct1 = svdot_s32(Ct1, ai2, bi0);
                    Ct1 = svdot_s32(Ct1, ai3, bi1);
                    Ct2 = svdot_s32(Ct2, ai4, bi0);
                    Ct2 = svdot_s32(Ct2, ai5, bi1);
                    Ct3 = svdot_s32(Ct3, ai6, bi0);
                    Ct3 = svdot_s32(Ct3, ai7, bi1);
                    Ct0 = svdot_s32(Ct0, ai0, bi2);
                    Ct0 = svdot_s32(Ct0, ai1, bi3);
                    Ct1 = svdot_s32(Ct1, ai2, bi2);
                    Ct1 = svdot_s32(Ct1, ai3, bi3);
                    Ct2 = svdot_s32(Ct2, ai4, bi2);
                    Ct2 = svdot_s32(Ct2, ai5, bi3);
                    Ct3 = svdot_s32(Ct3, ai6, bi2);
                    Ct3 = svdot_s32(Ct3, ai7, bi3);
                    Ct0 = svdot_s32(Ct0, ai0, bi4);
                    Ct0 = svdot_s32(Ct0, ai1, bi5);
                    Ct1 = svdot_s32(Ct1, ai2, bi4);
                    Ct1 = svdot_s32(Ct1, ai3, bi5);
                    Ct2 = svdot_s32(Ct2, ai4, bi4);
                    Ct2 = svdot_s32(Ct2, ai5, bi5);
                    Ct3 = svdot_s32(Ct3, ai6, bi4);
                    Ct3 = svdot_s32(Ct3, ai7, bi5);
                    Ct0 = svdot_s32(Ct0, ai0, bi6);
                    Ct0 = svdot_s32(Ct0, ai1, bi7);
                    Ct1 = svdot_s32(Ct1, ai2, bi6);
                    Ct1 = svdot_s32(Ct1, ai3, bi7);
                    Ct2 = svdot_s32(Ct2, ai4, bi6);
                    Ct2 = svdot_s32(Ct2, ai5, bi7);
                    Ct3 = svdot_s32(Ct3, ai6, bi6);
                    Ct3 = svdot_s32(Ct3, ai7, bi7);
                    // write Ct to Cr (Cr = "C reg", Ct = "C temp")
                    if (j == jj)
                    {
                        auto scalesA0 = svcvt_f32_f16_x(svptrue_b32(), scalesA);
                        auto scalesB0 = svcvt_f32_f16_x(svptrue_b32(), scalesB);
                        // scalesA0/B0, Cr0 = Ct*A0[0]*B0+Cr0
                        Cr0 = svmla_f32_z(svptrue_b32(), Cr0, svcvt_f32_s32_z(svptrue_b32(), Ct0),
                                          svmul_lane_f32(scalesB0, scalesA0, 0));
                        Cr1 = svmla_f32_z(svptrue_b32(), Cr1, svcvt_f32_s32_z(svptrue_b32(), Ct1),
                                          svmul_lane_f32(scalesB0, scalesA0, 1));
                        Cr2 = svmla_f32_z(svptrue_b32(), Cr2, svcvt_f32_s32_z(svptrue_b32(), Ct2),
                                          svmul_lane_f32(scalesB0, scalesA0, 2));
                        Cr3 = svmla_f32_z(svptrue_b32(), Cr3, svcvt_f32_s32_z(svptrue_b32(), Ct3),
                                          svmul_lane_f32(scalesB0, scalesA0, 3));
                    }
                    else
                    {
                        auto scalesA1 = svcvtlt_f32_f16_x(svptrue_b32(), scalesA);
                        auto scalesB1 = svcvtlt_f32_f16_x(svptrue_b32(), scalesB);
                        Cr0 = svmla_f32_z(svptrue_b32(), Cr0, svcvt_f32_s32_z(svptrue_b32(), Ct0),
                                          svmul_lane_f32(scalesB1, scalesA1, 0));
                        Cr1 = svmla_f32_z(svptrue_b32(), Cr1, svcvt_f32_s32_z(svptrue_b32(), Ct1),
                                          svmul_lane_f32(scalesB1, scalesA1, 1));
                        Cr2 = svmla_f32_z(svptrue_b32(), Cr2, svcvt_f32_s32_z(svptrue_b32(), Ct2),
                                          svmul_lane_f32(scalesB1, scalesA1, 2));
                        Cr3 = svmla_f32_z(svptrue_b32(), Cr3, svcvt_f32_s32_z(svptrue_b32(), Ct3),
                                          svmul_lane_f32(scalesB1, scalesA1, 3));
                    }
                }
            }
            svst1_f32(svptrue_b32(), C + i * N + k, Cr0);
            svst1_f32(svptrue_b32(), C + (i + 1) * N + k, Cr1);
            svst1_f32(svptrue_b32(), C + (i + 2) * N + k, Cr2);
            svst1_f32(svptrue_b32(), C + (i + 3) * N + k, Cr3);
        }
}
constexpr int NX = 8192;
alignas(16) int8_t tmp1[NX * NX];
alignas(16) int8_t tmp2[NX * NX];
alignas(16) float tmp3[NX * NX];
alignas(16) float16_t tmp4[NX * NX / 32];
alignas(16) float16_t tmp5[NX * NX / 32];
template <int S, int Z> void recurse(int A, int B, int C)
{
    if (S == Z)
        matmulX<S>(tmp1 + A, tmp2 + B, tmp4 + A / 32, tmp5 + B / 32, tmp3 + C);
    else
    {
        // C[0,0] += A[0,0] * B[0,0]
        recurse<S / 2, Z>(A, B, C);
        // C[0,1] += A[0,0] * B[0,1]
        recurse<S / 2, Z>(A, B + S * S / 4, C + S * S / 4);
        // C[0,0] += A[0,1] * B[1,0]
        recurse<S / 2, Z>(A + S * S / 4, B + 2 * S * S / 4, C);
        // C[0,1] += A[0,1] * B[1,1]
        recurse<S / 2, Z>(A + S * S / 4, B + 3 * S * S / 4, C + S * S / 4);
        // C[1,0] += A[1,0] * B[0,0]
        recurse<S / 2, Z>(A + 2 * S * S / 4, B, C + 2 * S * S / 4);
        // C[1,1] += A[1,0] * B[0,1]
        recurse<S / 2, Z>(A + 2 * S * S / 4, B + S * S / 4, C + 3 * S * S / 4);
        // C[1,0] += A[1,1] * B[1,0]
        recurse<S / 2, Z>(A + 3 * S * S / 4, B + 2 * S * S / 4, C + 2 * S * S / 4);
        // C[1,1] += A[1,1] * B[1,1]
        recurse<S / 2, Z>(A + 3 * S * S / 4, B + 3 * S * S / 4, C + 3 * S * S / 4);
    }
}
template <int NX> void matmultest()
{
    auto a = clock();
    recurse<8192, NX>(0, 0, 0);
    auto b = clock();
    cerr << NX << "," << 2. * 8192 * 8192 * 8192 / 1000 / (b - a) << endl;
}
void Yoshi()
{
    // This kernel is for 128-bit SVE only
    assert(svcntb() == 16);
    matmultest<128>();
    matmultest<128>();
    matmultest<256>();
    matmultest<512>();
    matmultest<1024>();
    matmultest<2048>();
    matmultest<4096>();
    matmultest<8192>();
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
