#include <bits/stdc++.h>
#include <immintrin.h>
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
void matmulX(float *A, float *B, float *C)
{
    A = (float *)__builtin_assume_aligned(A, 32);
    B = (float *)__builtin_assume_aligned(B, 32);
    C = (float *)__builtin_assume_aligned(C, 32);
    for (int i = 0; i < 32; i += 2)
    {
        __m256 Cr0 = _mm256_load_ps(C + i * 32);
        __m256 Cr1 = _mm256_load_ps(C + i * 32 + 8);
        __m256 Cr2 = _mm256_load_ps(C + i * 32 + 16);
        __m256 Cr3 = _mm256_load_ps(C + i * 32 + 24);
        __m256 Cr4 = _mm256_load_ps(C + i * 32 + 32);
        __m256 Cr5 = _mm256_load_ps(C + i * 32 + 40);
        __m256 Cr6 = _mm256_load_ps(C + i * 32 + 48);
        __m256 Cr7 = _mm256_load_ps(C + i * 32 + 56);
        for (int j = 0; j < 32; j++)
        {
            auto ai1 = _mm256_broadcast_ss(A + i * 32 + j);
            auto ai2 = _mm256_broadcast_ss(A + i * 32 + j + 32);
            auto bi1 = _mm256_load_ps(B + j * 32);
            auto bi2 = _mm256_load_ps(B + j * 32 + 8);
            auto bi3 = _mm256_load_ps(B + j * 32 + 16);
            auto bi4 = _mm256_load_ps(B + j * 32 + 24);
            // C[i] += A[i][j] * B[j]
            Cr0 = _mm256_fmadd_ps(ai1, bi1, Cr0);
            Cr1 = _mm256_fmadd_ps(ai1, bi2, Cr1);
            Cr2 = _mm256_fmadd_ps(ai1, bi3, Cr2);
            Cr3 = _mm256_fmadd_ps(ai1, bi4, Cr3);
            Cr4 = _mm256_fmadd_ps(ai2, bi1, Cr4);
            Cr5 = _mm256_fmadd_ps(ai2, bi2, Cr5);
            Cr6 = _mm256_fmadd_ps(ai2, bi3, Cr6);
            Cr7 = _mm256_fmadd_ps(ai2, bi4, Cr7);
        }
        _mm256_store_ps(C + i * 32, Cr0);
        _mm256_store_ps(C + i * 32 + 8, Cr1);
        _mm256_store_ps(C + i * 32 + 16, Cr2);
        _mm256_store_ps(C + i * 32 + 24, Cr3);
        _mm256_store_ps(C + i * 32 + 32, Cr4);
        _mm256_store_ps(C + i * 32 + 40, Cr5);
        _mm256_store_ps(C + i * 32 + 48, Cr6);
        _mm256_store_ps(C + i * 32 + 56, Cr7);
    }
}
constexpr int NX = 2048;
alignas(32) float tmp1[NX * NX];
alignas(32) float tmp2[NX * NX];
alignas(32) float tmp3[NX * NX];
alignas(32) float A_test[NX * NX];
alignas(32) float B_test[NX * NX];
alignas(32) float C_test[NX * NX];
template <int S> void recurse(float *A, float *B, float *C)
{
    A = (float *)__builtin_assume_aligned(A, 32);
    B = (float *)__builtin_assume_aligned(B, 32);
    C = (float *)__builtin_assume_aligned(C, 32);
    if (S == 32)
        matmulX(A, B, C);
    else
    {
        // C[0,0] += A[0,0] * B[0,0]
        recurse<S / 2>(A, B, C);
        // C[0,1] += A[0,0] * B[0,1]
        recurse<S / 2>(A, B + S * S / 4, C + S * S / 4);
        // C[1,0] += A[1,0] * B[0,0]
        recurse<S / 2>(A + 2 * S * S / 4, B, C + 2 * S * S / 4);
        // C[1,1] += A[1,0] * B[0,1]
        recurse<S / 2>(A + 2 * S * S / 4, B + S * S / 4, C + 3 * S * S / 4);
        // C[0,0] += A[0,1] * B[1,0]
        recurse<S / 2>(A + S * S / 4, B + 2 * S * S / 4, C);
        // C[0,1] += A[0,1] * B[1,1]
        recurse<S / 2>(A + S * S / 4, B + 3 * S * S / 4, C + S * S / 4);
        // C[1,0] += A[1,1] * B[1,0]
        recurse<S / 2>(A + 3 * S * S / 4, B + 2 * S * S / 4, C + 2 * S * S / 4);
        // C[1,1] += A[1,1] * B[1,1]
        recurse<S / 2>(A + 3 * S * S / 4, B + 3 * S * S / 4, C + 3 * S * S / 4);
    }
}
// repack src from row-major to tree-form
template <int N> void repackA(float *__restrict__ dest, float *__restrict__ src, int tl, int tr)
{
    src = (float *)__builtin_assume_aligned(src, 32);
    dest = (float *)__builtin_assume_aligned(dest, 32);
    if (N == 32)
    {
        // copy, row-order
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                dest[i * 32 + j] = src[(tl + i) * NX + tr + j];
    }
    else
    {
        repackA<N / 2>(dest, src, tl, tr);
        repackA<N / 2>(dest + N * N / 4, src, tl, tr + N / 2);
        repackA<N / 2>(dest + 2 * N * N / 4, src, tl + N / 2, tr);
        repackA<N / 2>(dest + 3 * N * N / 4, src, tl + N / 2, tr + N / 2);
    }
}
template <int N> void repackC(float *__restrict__ dest, float *__restrict__ src, int tl, int tr)
{
    src = (float *)__builtin_assume_aligned(src, 32);
    dest = (float *)__builtin_assume_aligned(dest, 32);
    if (N == 32)
    {
        // copy, still row-order
        for (int i = 0; i < 32; i++)
            for (int j = 0; j < 32; j++)
                dest[(tl + i) * NX + tr + j] = src[i * 32 + j];
    }
    else
    {
        repackC<N / 2>(dest, src, tl, tr);
        repackC<N / 2>(dest, src + N * N / 4, tl, tr + N / 2);
        repackC<N / 2>(dest, src + 2 * N * N / 4, tl + N / 2, tr);
        repackC<N / 2>(dest, src + 3 * N * N / 4, tl + N / 2, tr + N / 2);
    }
}
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
void Yoshi()
{
    auto a = clock();
    for (int i = 0; i < 10; i++)
    {
        repackA<NX>(tmp1, A_test, 0, 0);
        repackA<NX>(tmp2, B_test, 0, 0);
        recurse<NX>(tmp1, tmp2, tmp3);
        repackC<NX>(C_test, tmp3, 0, 0);
    }
    auto b = clock();
    cout << 10.0 * NX * NX * NX * 2 / (b - a) << endl;
}
signed main()
{
#ifndef yoshi_likes_e4
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
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