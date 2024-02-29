// nvcc -O3 -g -lineinfo block_wrap_reduction.cu
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

constexpr int N{10000};
constexpr int BLOCKS{2*3};//this number is hardware-dependent; usually #SM*2 is a good number.
constexpr int THREADS{1024};
constexpr int WRAP_SIZE{32};
constexpr int WRAPS_PER_BLOCK{THREADS/WRAP_SIZE};
constexpr unsigned int MASK{0xffffffff};

__device__ float warpReduce(volatile float* shArr) { 
    int idx = threadIdx.x % WRAP_SIZE; //the lane index in the warp
    if (idx<16) {
      shArr[idx] += shArr[idx+16];
      shArr[idx] += shArr[idx+8];
      shArr[idx] += shArr[idx+4];
      shArr[idx] += shArr[idx+2];
      shArr[idx] += shArr[idx+1];
    }
    return shArr[0];
}


__device__ float warpReduce2(float value) {
    // for (int i=16; i > 0; i >>= 1)
    //     value += __shfl_down_sync(MASK, value, i);
    value += __shfl_down_sync(MASK, value, 16);
    value += __shfl_down_sync(MASK, value, 8);
    value += __shfl_down_sync(MASK, value, 4);
    value += __shfl_down_sync(MASK, value, 2);
    value += __shfl_down_sync(MASK, value, 1);

    return value;
}


__device__ float warpReduce3(float value) {
    // Use XOR mode to perform butterfly reduction
    for (int i=16; i > 0; i >>= 1)
        value += __shfl_xor_sync(MASK, value, i);

    return value;
}


__device__ bool lastBlock(int* counter) {
    __threadfence(); //ensure that partial result is visible by all BLOCKS
    int last = 0;
    if (threadIdx.x == 0)
        last = atomicAdd(counter, 1);
    return __syncthreads_or(last == gridDim.x-1);
}


__global__ void reduce_tree(int N, float* __restrict__ inVec, float* __restrict__ partial, int* __restrict__ lastBlockCounter) {
    int thIdx = threadIdx.x;
    int globalIdx = thIdx + blockIdx.x * blockDim.x;
    const int gridSize = blockDim.x * gridDim.x;

    //perform private sums
    float sum{0};
    for (int i = globalIdx; i < N; i += gridSize)
        sum += inVec[i];

    // share among block threads private sum
    __shared__ float shArr[THREADS];
    shArr[thIdx] = sum;
    __syncthreads();

    // tree reduction among block threads
    for (int size = blockDim.x/2; size > 0; size >>= 1) { //uniform
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }

    // each block share its partial reduction
    if (thIdx == 0)
        partial[blockIdx.x] = shArr[0];

    // choose last block to perform the final reduction
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx < gridSize ? partial[thIdx] : 0;
        __syncthreads();

        // tree reduction among last block threads
        for (int size = blockDim.x/2; size > 0; size >>= 1) { //uniform
            if (thIdx < size)
                shArr[thIdx] += shArr[thIdx+size];
            __syncthreads();
        }

        if (thIdx == 0)
            partial[0] = shArr[0];
    }
}


__global__ void reduce_usm(int N, float* __restrict__ inVec, float* __restrict__ partial, int* __restrict__ lastBlockCounter) {
    int thIdx = threadIdx.x;
    int globalIdx = thIdx + blockIdx.x * blockDim.x;
    const int gridSize = blockDim.x * gridDim.x;

    //perform private sums
    float sum = 0;
    for (int i = globalIdx; i < N; i += gridSize)
        sum += inVec[i];

    // share among block threads private sum
    __shared__ float shArr[THREADS];
    shArr[thIdx] = sum;
    // SIMT reduction
    warpReduce(&shArr[thIdx & ~(WRAP_SIZE-1)]); // &shArr[thIdx & ~(WRAP_SIZE-1)] = r + warpIdx*32
    __syncthreads();

    //first warp only
    if (thIdx < WRAP_SIZE) {
        shArr[thIdx] = thIdx * WRAP_SIZE < blockDim.x ? shArr[thIdx*WRAP_SIZE] : 0;
        //join the other wrap reductions
        warpReduce(shArr);
        
        // each block share its partial reduction
        if (thIdx == 0)
            partial[blockIdx.x] = shArr[0];
    }

    // choose last block to perform the final reduction
    if (lastBlock(lastBlockCounter)) {
        shArr[thIdx] = thIdx < gridSize ? partial[thIdx] : 0;
        //shArr[thIdx] = warpReduce2(shArr[thIdx]);
        warpReduce(&shArr[thIdx & ~(WRAP_SIZE-1)]);
        __syncthreads();

        //first warp only
        if (thIdx < WRAP_SIZE) {
            shArr[thIdx] = thIdx * WRAP_SIZE < blockDim.x ? shArr[thIdx*WRAP_SIZE] : 0;
            //join the other wrap reductions
            warpReduce(shArr);
            
            // each block share its partial reduction
            if (thIdx == 0)
                partial[0] = shArr[0];
        }
    }
}


__global__ void reduce_register(int N, float* __restrict__ inVec, float* __restrict__ partial, int* __restrict__ lastBlockCounter) {
    int thIdx = threadIdx.x;
    int globalIdx = thIdx + blockIdx.x * blockDim.x;
    const int gridSize = blockDim.x * gridDim.x;
    const int wrapIdx = thIdx / WRAPS_PER_BLOCK;

    //perform private sums
    float sum{0};
    for (int i = globalIdx; i < N; i += gridSize)
        sum += inVec[i];

    // share among block threads private sum
    __shared__ float shArr[WRAPS_PER_BLOCK];
    // SIMT reduction
    shArr[wrapIdx] = warpReduce2(sum);
    __syncthreads();

    //first warp only
    if (thIdx < WRAP_SIZE) {
        sum = thIdx * WRAP_SIZE < blockDim.x ? shArr[thIdx] : 0;
        //join the other wrap reductions
        sum = warpReduce2(sum);
        
        // each block shares its partial reduction
        if (thIdx == 0)
            partial[blockIdx.x] = sum;
    }

    // choose last block to perform the final reduction
    if (lastBlock(lastBlockCounter)) {
        sum = thIdx < gridSize ? partial[thIdx] : 0;
        shArr[wrapIdx] = warpReduce2(sum);
        __syncthreads();

        //first warp only
        if (thIdx < WRAP_SIZE) {
            sum = thIdx * WRAP_SIZE < blockDim.x ? shArr[thIdx] : 0;
            //join the other wrap reductions
            sum = warpReduce2(sum);
            
            if (thIdx == 0)
                partial[0] = sum;
        }
    }
}


int main() {
    // init vec
    auto times_two = [] (int i) {return i*2.0f;};
    std::vector<float> vec(N);
    std::iota(vec.begin(), vec.end(), 0);
    std::transform(vec.begin(), vec.end(), vec.begin(), times_two);

    float host_sum{0}, tree_sum{0}, usm_sum{0}, reg_sum{0};
    std::for_each(vec.begin(), vec.end(), [&host_sum](auto n){
        host_sum += n;
    });
    
    float *d_vec, *d_partial; 
    int* lastBlockCounter;

    cudaMalloc(&d_vec, N*sizeof(float));
    cudaMalloc(&d_partial, BLOCKS*sizeof(float));
    cudaMalloc(&lastBlockCounter, BLOCKS*sizeof(int));
    cudaMemcpy(d_vec, vec.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_partial, 0, BLOCKS*sizeof(float));
    cudaMemset(lastBlockCounter, 0, BLOCKS*sizeof(int));
    cudaDeviceSynchronize();

    double t_red_tree{0.f}, t_red_usm{0.f}, t_red_reg{0.f};
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start, 0);
    // cudaEventCreate(&stop, 0);

    // cudaEventRecord(start);
    auto start = std::chrono::high_resolution_clock::now();
    reduce_tree<<<BLOCKS, THREADS>>>(N, d_vec, d_partial, lastBlockCounter);
    cudaDeviceSynchronize();
    // cudaEventRecord(stop);
    // cudaEventElapsedTime(&t_red_tree, start, stop);
    t_red_tree = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    cudaMemcpy(&tree_sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //clean memory
    cudaMemset(d_partial, 0, BLOCKS*sizeof(float));
    cudaMemset(lastBlockCounter, 0, BLOCKS*sizeof(float));
    cudaDeviceSynchronize();

    // cudaEventRecord(start);
    start = std::chrono::high_resolution_clock::now();
    reduce_usm<<<BLOCKS, THREADS>>>(N, d_vec, d_partial, lastBlockCounter);
    cudaDeviceSynchronize();
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&t_red_usm, start, stop);
    t_red_usm = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    cudaMemcpy(&usm_sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //clean memory
    cudaMemset(d_partial, 0, BLOCKS*sizeof(float));
    cudaMemset(lastBlockCounter, 0, BLOCKS*sizeof(float));
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    reduce_register<<<BLOCKS, THREADS>>>(N, d_vec, d_partial, lastBlockCounter);
    cudaDeviceSynchronize();
    t_red_reg = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
    cudaMemcpy(&reg_sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "Host sum      = " << host_sum << std::endl
              << "Tree sum      = " << tree_sum << ", in " << t_red_tree << " us" << std::endl
              << "USM sum       = " << usm_sum << ", in " << t_red_usm << " us" << std::endl
              << "Register sum  = " << reg_sum << ", in " << t_red_reg << " us" << std::endl;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    cudaFree(d_vec);
    cudaFree(d_partial);
    cudaFree(lastBlockCounter);

    return 0;
}