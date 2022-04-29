#include "cuda_utils.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

const int BLOCK_SIZE = 32;
int validate_ref = 1;
int debug = 0;
xtimer_t dgemm_ref_timer;
xtimer_t dgemm_timer;

// Computes C = A*B, where A is a M by K matrix, B is a K by N matrix, C is a M by N matrix.
// Matrices are stored in row-major order.
static void ref_mm(const double *A, const double *B, double *C, const int M, const int N, const int K) {
    int i, j, k;
    double sum;
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.;
            for (k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

__global__ void gpu_mm(const double *A, const double *B,
                            double *C,
                            const int M, const int N, const int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

static void rand_init(double *data, const int N) {
    for (int i = 0; i < N; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
}

// static void print_mat(const double *a, const int M, const int N) {
//     printf("---------------\n");
//     for (int r = 0; r < M; r++) {
//         for (int c = 0; c < N; c++) {
//             printf("%.4lf, ", a[r * N + c]);
//         }
//         printf("\n");
//     }
// }

static bool is_matching(const double *a, const double *b, const int N) {
    if (validate_ref){
        for (int i = 0; i < N; i++) {
            double diff = fabs(a[i] - b[i]);
            if (diff > 1e-8) { // double has just 6.5 significant digits
                printf("Mismatch at %d: %lf, %lf (%lf)\n", i, a[i], b[i], diff);
                return false;
            }
        }
    }
    return true;
}

static bool test_mm(int M, int N, int K) {
    bool status = true;
    double *a = (double *)malloc(M * K * sizeof(double));
    double *b = (double *)malloc(K * N * sizeof(double));
    double *c = (double *)calloc(M * N, sizeof(double));
    double *c_ref = (double *)calloc(M * N, sizeof(double));

    // device pointers
    double *dev_a, *dev_b, *dev_c;
    checkCudaErrors(cudaMalloc((void **)&dev_a, M * K * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_b, K * N * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **)&dev_c, M * N * sizeof(double)));

    rand_init(a, M * K);
    rand_init(b, K * N);

    checkCudaErrors(cudaMemcpy(dev_a, a, M * K * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, K * N * sizeof(double), cudaMemcpyHostToDevice));

    // Reference implementation
    if (validate_ref){
        timer_start(&dgemm_ref_timer);
        ref_mm(a, b, c_ref, M, N, K);
        timer_stop(&dgemm_ref_timer);
        timer_elapsed_time(&dgemm_ref_timer);
    }
    // Evaluated implementation
    timer_start(&dgemm_timer);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    gpu_mm <<< grid, block>>>(dev_a, dev_b, dev_c, M, N, K);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // gpu_mm(dev_a, dev_b, dev_c, M, N, K);
    checkCudaErrors(cudaMemcpy(c, dev_c, M * N * sizeof(double), cudaMemcpyDeviceToHost));

    timer_stop(&dgemm_timer);
    timer_elapsed_time(&dgemm_timer);

    status = is_matching(c, c_ref, M * N);
    // if (debug){
    //     printf("Referece Matrix:\n");
    //     print_mat(c_ref, M, N);
    //     printf("Calculated Matrix:\n");
    //     print_mat(c, M, N);
    // }
    // Cleanup
    checkCudaErrors(cudaFree(dev_a));
    checkCudaErrors(cudaFree(dev_b));
    checkCudaErrors(cudaFree(dev_c));
    free(a);
    free(b);
    free(c);
    free(c_ref);
    return status;
}

int main(int argc, char **argv) {
    int M = 1<<9; 
    int N = M;
    int K = M;
    // int N = 1<<10;
    // int K = 1<<11;
    validate_ref = 1;
    debug = 0;
    if(argc > 1){
        M = atoi(argv[1]);        
    }
    // if(argc > 2){
    //     N = atoi(argv[2]);        
    // }
    // if(argc > 3){
    //     K = atoi(argv[3]);        
    // }
    if(argc > 2){
        validate_ref = atoi(argv[4]);        
    }
    // if(argc > 3){
    //     debug = atoi(argv[5]);
    // }
    printf("M:%d, validate:%d\n", M, validate_ref);

    timer_clear(&dgemm_timer);
    timer_clear(&dgemm_ref_timer);

    bool status;
    printf("Testing Matrix Multiply... ");
    status = test_mm(M, N, K);
    if (status) {
        printf("PASS\n");
    } else {
        printf("FAILED\n");
    }
    printf("MM CPU time: %.6lf\n", timer_elapsed_time(&dgemm_ref_timer));
    printf("MM GPU time: %.6lf\n", timer_elapsed_time(&dgemm_timer));

    return 0;
}