#include "cuda.h"
#include <cuda_runtime.h>

// Make a struct to hold the parameters for the kernel
template<typename T=double>
struct powParams {
    int n;
    T *argx;
    T *argy;
    T *res;
};

template<typename T=double>
struct powintParams {
    int n;
    T *argx;
    int *argy;
    T *res;
};

// Tait-Murnaghan Equation of State
// https://en.wikipedia.org/wiki/Murnaghan_equation_of_state
// Traditional implementation using pow() function and J
template<typename T=double>
__global__ void traditional_murnaghan_eos (powParams<T> parms)
{

    int i;

    int totalThreads = gridDim.x * blockDim.x;

    int ctaStart = blockDim.x * blockIdx.x;

    for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads) {

#if 0

        parms.res[i] = parms.argx[i] * parms.argy[i];      // baseline

#elif 0

        parms.res[i] = pow (parms.argx[i], 7.0);           // exponent compile-time constant   

#else
        const T P0 = static_cast<T>(101000); // Ambient pressure in Pa
        const T bulk_gamma = static_cast<T>(2.2e9) / parms.argy[i] ; // Bulk modulus in in Pa over bulk modulus derivative w.r.t. pressure
        const T one = static_cast<T>(1.0);
        parms.res[i] = bulk_gamma * ( pow (parms.argx[i], -parms.argy[i]) - one) + P0; // exponent variable

#endif

    }

}

// Bonus 2023 - Reformulated Tait-Murnaghan Equation of State
// https://en.wikipedia.org/wiki/Murnaghan_equation_of_state
// Optimized implementation using expm1() log1p() function and j=(1-J)
template<typename T=double>
__global__ void bonus_murnaghan_eos (powParams<T> parms)
{

    int i;

    int totalThreads = gridDim.x * blockDim.x;

    int ctaStart = blockDim.x * blockIdx.x;

    for (i = ctaStart + threadIdx.x; i < parms.n; i += totalThreads) {

#if 0

        parms.res[i] = parms.argx[i] * parms.argy[i];      // baseline

#elif 0

        parms.res[i] = pow (parms.argx[i], 7.0);           // exponent compile-time constant   

#else
        const T P0 = static_cast<T>(101000); // Ambient pressure in Pa
        const T bulk_gamma = static_cast<T>(2.2e9) / parms.argy[i] ; // Bulk modulus in in Pa over bulk modulus derivative w.r.t. pressure
        // parms.res[i] = bulk_gamma*expm1(-parms.argy[i] * log1p(-parms.argx[i])) + P0; // exponent variable
        parms.res[i] = fma(bulk_gamma, expm1(-parms.argy[i] * log1p(-parms.argx[i])), P0); // exponent variable

#endif

    }

}


// Study register usage of Tait-Murnaghan Equation of State in CUDA
// https://en.wikipedia.org/wiki/Murnaghan_equation_of_state
// Traditional implementation using pow() function and J
// Optimized implementation using expm1() log1p() function and j=(1-J)
// Justin Bonus 2023 - University of Washington, Seattle, Civil and Environmental Engineering
// Compile with:
// nvcc -std=c++11 -arch=sm_61 -dc -Xptxas="-v" registers.cu
// Replace -arch=sm_61 with your GPU architecture, e.g. sm_75 for 2080ti, or sm_80 for A100
// Register use varies with GPU architecture, so you may need to adjust threads per block
// Typically register use is:
// Double-Precision
// Traditional:       28 registers
// Bonus 2023:        24 registers
// Single-Precision
// Traditional:       20 registers
// Bonus 2023:        18 registers
int main() {
    // Parameters
    const int n = 10000;
    const int size_d = n * sizeof(double);
    const int size_f = n * sizeof(float);

    // Allocate memory on the host
    double *h_argx_d = (double*)malloc(size_d);
    double *h_argy_d = (double*)malloc(size_d);
    double *h_res_d = (double*)malloc(size_d);
    float *h_argx_f = (float*)malloc(size_f);
    float *h_argy_f = (float*)malloc(size_f);
    float *h_res_f = (float*)malloc(size_f);

    // Initialize the input data
    for (int i = 0; i < n; i++) {
        h_argx_d[i] = i;
        h_argy_d[i] = i;
        h_argx_f[i] = i;
        h_argy_f[i] = i;
    }

    // Allocate memory on the device
    double *d_argx_d, *d_argy_d, *d_res_d;
    cudaMalloc((void**)&d_argx_d, size_d);
    cudaMalloc((void**)&d_argy_d, size_d);
    cudaMalloc((void**)&d_res_d, size_d);
    float *d_argx_f, *d_argy_f, *d_res_f;
    cudaMalloc((void**)&d_argx_f, size_f);
    cudaMalloc((void**)&d_argy_f, size_f);
    cudaMalloc((void**)&d_res_f, size_f);

    // Copy the input data to the device
    cudaMemcpy(d_argx_d, h_argx_d, size_d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_argy_d, h_argy_d, size_d, cudaMemcpyHostToDevice);

    cudaMemcpy(d_argx_f, h_argx_f, size_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_argy_f, h_argy_f, size_f, cudaMemcpyHostToDevice);

    // Create and initialize the parameters
    powParams<double> params_d;
    powParams<float> params_f;

    params_d.n = n;
    params_d.argx = d_argx_d;
    params_d.argy = d_argy_d;
    params_d.res = d_res_d;
    params_f.n = n;
    params_f.argx = d_argx_f;
    params_f.argy = d_argy_f;
    params_f.res = d_res_f;
    // Specify the execution configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (params_d.n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the traditional kernel
    traditional_murnaghan_eos<<<blocksPerGrid, threadsPerBlock>>>(params_f);
    traditional_murnaghan_eos<<<blocksPerGrid, threadsPerBlock>>>(params_d);
    // Launch the reformulated kernel

    bonus_murnaghan_eos<<<blocksPerGrid, threadsPerBlock>>>(params_f);
    bonus_murnaghan_eos<<<blocksPerGrid, threadsPerBlock>>>(params_d);
    

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_res_d, d_res_d, size_d, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_res_f, d_res_f, size_f, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_argx_d);
    cudaFree(d_argy_d);
    cudaFree(d_res_d);
    cudaFree(d_argx_f);
    cudaFree(d_argy_f);
    cudaFree(d_res_f);


    // Free host memory
    free(h_argx_d);
    free(h_argy_d);
    free(h_res_d);
    free(h_argx_f);
    free(h_argy_f);
    free(h_res_f);
    return 0;
}