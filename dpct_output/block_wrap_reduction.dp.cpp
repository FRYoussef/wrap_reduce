// clang++ -O3 -w -fsycl -fsycl-targets=nvptx64-nvidia-cuda block_wrap_reduction.dp.cpp 
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

constexpr int N{100000000};
constexpr int BLOCKS{2*6};//this number is hardware-dependent; usually #SM*2 is a good number.
constexpr int THREADS{1024};
constexpr int WRAP_SIZE{32};

// CUDA GPU selector
class CudaGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string DriverVersion = Device.get_info<sycl::info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

class IntelGPUSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};


int warpReduce(volatile int* shArr, sycl::nd_item<3> item_ct1) {
    int idx = item_ct1.get_local_id(2) % WRAP_SIZE; // the lane index in the
                                                    // warp
    if (idx<16) {
      shArr[idx] += shArr[idx+16];
      shArr[idx] += shArr[idx+8];
      shArr[idx] += shArr[idx+4];
      shArr[idx] += shArr[idx+2];
      shArr[idx] += shArr[idx+1];
    }
    return shArr[0];
}


int warpReduce2(int value, sycl::nd_item<3> item_ct1) {
    for (int i=16; i > 0; i >>= 1)
        value += sycl::shift_group_left(item_ct1.get_sub_group(), value, i);

    return value;
}


int warpReduce3(int& value, sycl::nd_item<3> item_ct1) {
    // Use XOR mode to perform butterfly reduction
    for (int i=1; i < 32; i <<= 1)
        value += sycl::permute_group_by_xor(item_ct1.get_sub_group(), value, i);

    return value;
}


bool lastBlock(int* counter, sycl::nd_item<3> item_ct1) {
    sycl::atomic_fence(
        sycl::ext::oneapi::memory_order::acq_rel,
        sycl::ext::oneapi::memory_scope::device); // ensure that partial result
                                                  // is visible by all BLOCKS
    int last = 0;
    if (item_ct1.get_local_id(2) == 0)
        last = sycl::atomic<int>(sycl::global_ptr<int>(counter)).fetch_add(1);

    return (item_ct1.barrier(sycl::access::fence_space::local_space),
            sycl::any_of_group(item_ct1.get_group(),
                               last == item_ct1.get_group_range(2) - 1));
}


void reduce(int N, int* __restrict__ inVec, int* __restrict__ partial, int* __restrict__ lastBlockCounter,
            sycl::nd_item<3> item_ct1, int *shArr) {
    int thIdx = item_ct1.get_local_id(2);
    int globalIdx = thIdx + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int gridSize = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    //perform private sums
    int sum{0};
    for (int i = globalIdx; i < N; i += gridSize)
        sum += inVec[i];

    // share among block threads private sum
    shArr[thIdx] = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // tree reduction among block threads
    for (int size = item_ct1.get_local_range(2) / 2; size > 0; size >>= 1) {
        if (thIdx < size)
            shArr[thIdx] += shArr[thIdx+size];

        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    // each block share its partial reduction
    if (thIdx == 0)
        partial[item_ct1.get_group(2)] = shArr[0];

    // choose last block to perform the final reduction
    if (lastBlock(lastBlockCounter, item_ct1)) {
        shArr[thIdx] = thIdx < gridSize ? partial[thIdx] : 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // tree reduction among last block threads
        for (int size = item_ct1.get_local_range(2) / 2; size > 0;
             size >>= 1) { // uniform
            if (thIdx < size)
                shArr[thIdx] += shArr[thIdx+size];

            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

        if (thIdx == 0)
            partial[0] = shArr[0];
    }
}


void reduce2(int N, int* __restrict__ inVec, int* __restrict__ partial, int* __restrict__ lastBlockCounter,
             sycl::nd_item<3> item_ct1, int *shArr) {
    int thIdx = item_ct1.get_local_id(2);
    int globalIdx = thIdx + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const int gridSize = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    //perform private sums
    int sum = 0;
    for (int i = globalIdx; i < N; i += gridSize)
        sum += inVec[i];

    // share among block threads private sum
    shArr[thIdx] = sum;
    // SIMT reduction
    //warpReduce(&shArr[thIdx & ~(WRAP_SIZE-1)]); // &shArr[thIdx & ~(WRAP_SIZE-1)] = r + warpIdx*32
    shArr[thIdx] = warpReduce2(shArr[thIdx], item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    //first warp only
    if (thIdx < WRAP_SIZE) {
        shArr[thIdx] = thIdx * WRAP_SIZE < item_ct1.get_local_range(2) ? shArr[thIdx * WRAP_SIZE] : 0;
        //join the other wrap reductions
        //warpReduce(shArr);
        shArr[thIdx] = warpReduce2(shArr[thIdx], item_ct1);

        // each block share its partial reduction
        if (thIdx == 0)
            partial[item_ct1.get_group(2)] = shArr[0];
    }

    // choose last block to perform the final reduction
    if (lastBlock(lastBlockCounter, item_ct1)) {
        shArr[thIdx] = thIdx < gridSize ? partial[thIdx] : 0;
        shArr[thIdx] = warpReduce2(shArr[thIdx], item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        //first warp only
        if (thIdx < WRAP_SIZE) {
            shArr[thIdx] = thIdx * WRAP_SIZE < item_ct1.get_local_range(2)
                               ? shArr[thIdx * WRAP_SIZE]
                               : 0;
            //join the other wrap reductions
            shArr[thIdx] = warpReduce2(shArr[thIdx], item_ct1);

            // each block share its partial reduction
            if (thIdx == 0)
                partial[0] = shArr[0];
        }
    }
}

int main() {
    CudaGpuSelector device{};;
    sycl::queue queue{device, sycl::property::queue::enable_profiling{}};
    std::cout << "Running on \"" << queue.get_device().get_info<sycl::info::device::name>() << "\" under SYCL." << std::endl;

    // init vec
    auto times_two = [] (int i) {return i*2;};
    std::vector<int> vec(N);
    std::iota(vec.begin(), vec.end(), 0);
    std::transform(vec.begin(), vec.end(), vec.begin(), times_two);

    int host_sum{0}, block_sum{0}, wrap_sum{0};
    std::for_each(vec.begin(), vec.end(), [&host_sum](auto n){
        host_sum += n;
    });
    
    int *d_vec, *d_partial, *lastBlockCounter;

    d_vec = sycl::malloc_device<int>(N, queue);
    d_partial = sycl::malloc_device<int>(BLOCKS, queue);
    lastBlockCounter = sycl::malloc_device<int>(BLOCKS, queue);
    queue.memcpy(d_vec, vec.data(), N * sizeof(int));
    queue.memset(d_partial, 0, BLOCKS * sizeof(int));
    queue.memset(lastBlockCounter, 0, BLOCKS * sizeof(int));
    queue.wait_and_throw();

    double block_time{0}, wrap_time{0};
    sycl::event ev;

    ev = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shArr_acc_ct1(sycl::range<1>(THREADS), cgh);

        auto N_ct0 = N;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS) *
                                               sycl::range<3>(1, 1, THREADS),
                                           sycl::range<3>(1, 1, THREADS)),
                         [=](sycl::nd_item<3> item_ct1) {
                             reduce(N_ct0, d_vec, d_partial, lastBlockCounter,
                                    item_ct1, shArr_acc_ct1.get_pointer());
                         });
    });
    block_time = ev.get_profiling_info<sycl::info::event_profiling::command_end>()
        - ev.get_profiling_info<sycl::info::event_profiling::command_start>();
    queue.memcpy(&block_sum, d_partial, sizeof(int)).wait();

    //clean memory
    queue.memset(d_partial, 0, BLOCKS * sizeof(int));
    queue.memset(lastBlockCounter, 0, BLOCKS * sizeof(int)).wait();


    ev = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shArr_acc_ct1(sycl::range<1>(THREADS), cgh);

        auto N_ct0 = N;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS) *
                                  sycl::range<3>(1, 1, THREADS),
                              sycl::range<3>(1, 1, THREADS)),
            [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                reduce2(N_ct0, d_vec, d_partial, lastBlockCounter, item_ct1,
                        shArr_acc_ct1.get_pointer());
            });
    });
    wrap_time = ev.get_profiling_info<sycl::info::event_profiling::command_end>()
        - ev.get_profiling_info<sycl::info::event_profiling::command_start>();;
    queue.memcpy(&wrap_sum, d_partial, sizeof(int)).wait();

    std::cout << "Host sum  = " << host_sum << std::endl
              << "Block sum = " << block_sum << ", in " << block_time << " ns" << std::endl
              << "Wrap sum  = " << wrap_sum << ", in " << wrap_time << " ns" << std::endl;

    sycl::free(d_vec, queue);
    sycl::free(d_partial, queue);
    sycl::free(lastBlockCounter, queue);

    return 0;
}