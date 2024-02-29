// dpcpp -qopenmp -DCPU_DEVICE -I${ONEAPI_ROOT}/dpcpp-ct/latest/include/ atomic_test.cpp

#include <chrono>
#include <iostream>
#include <omp.h>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

using namespace sycl;

#define TYPE unsigned int
static constexpr int N = 100000;

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

// Intel GPU
class IntelGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};


void run_sycl_atomic() {
    TYPE h_sum{0};
    TYPE* values;
    TYPE* sum;
#if defined(INTEL_GPU_DEVICE)
	IntelGpuSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CudaGpuSelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
    values = malloc_device<TYPE>(N * sizeof(TYPE), queue);
    sum    = malloc_device<TYPE>(sizeof(TYPE), queue);

    const int CUs          = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    const int attrs_per_CU = N / CUs;
    const int remaining    = N % CUs;

    queue.memset(values, 1, N * sizeof(TYPE));
    queue.memset(sum, 0, sizeof(TYPE));
    queue.wait();
    const auto start1 = std::chrono::high_resolution_clock::now();
    queue.submit([&](handler &h) {
        h.parallel_for<class sycl_red>(nd_range(range(CUs), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == CUs-1) ? attrs_per_CU + remaining : attrs_per_CU;

            for (size_t i = offset; i < offset + length; i++)
                sycl::atomic<TYPE>(sycl::global_ptr<TYPE>(&sum[0])).fetch_add(values[i]);
        });
    });
    queue.wait();
    const auto end1      = std::chrono::high_resolution_clock::now();
    const auto duration1 = std::chrono::duration_cast<std::chrono::duration<float>>(end1 - start1);
    queue.memcpy(&h_sum, sum, sizeof(TYPE));
    queue.wait();
    std::cout << "SYCL atomic      -> SUM = " << h_sum << ", in " << duration1.count() << "s" << std::endl;

    free(values, queue);
    free(sum, queue);
}


void run_dpct_atomic() {
    TYPE h_sum{0};
    TYPE* values;
    TYPE* sum;
#if defined(INTEL_GPU_DEVICE)
	IntelGpuSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CudaGpuSelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
    values = malloc_device<TYPE>(N * sizeof(TYPE), queue);
    sum    = malloc_device<TYPE>(sizeof(TYPE), queue);

    const int CUs          = queue.get_device().get_info<sycl::info::device::max_compute_units>();
    const int attrs_per_CU = N / CUs;
    const int remaining    = N % CUs;

    queue.memset(values, 1, N * sizeof(TYPE));
    queue.memset(sum, 0, sizeof(TYPE));
    queue.wait();
    const auto start2 = std::chrono::high_resolution_clock::now();
    queue.submit([&](handler &h) {
        h.parallel_for<class dpct_red>(nd_range(range(CUs), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == CUs-1) ? attrs_per_CU + remaining : attrs_per_CU;

            for (size_t i = offset; i < offset + length; i++)
                dpct::atomic_fetch_add(&sum[0], values[i]);
        });
    });
    queue.wait();
    const auto end2      = std::chrono::high_resolution_clock::now();
    const auto duration2 = std::chrono::duration_cast<std::chrono::duration<float>>(end2 - start2);
    queue.memcpy(&h_sum, sum, sizeof(TYPE));
    queue.wait();
    std::cout << "DPCT atomic      -> SUM = " << h_sum << ", in " << duration2.count() << "s" << std::endl;

    free(values, queue);
    free(sum, queue);
}


void run_on_omp() {
    TYPE sum = 0;
    TYPE* values;
    values = new TYPE[N];
    std::memset(values, 1, N * sizeof(TYPE));

    const auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (size_t i = 0; i < N; i++) {
        sum += values[i];
    }
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cout << "OpenMP reduction -> SUM = " << sum << ", in " << duration.count() << "s" << std::endl;

    delete[] values;
}


int main(int argc, const char* argv[]) { 
    run_dpct_atomic();
    run_sycl_atomic();
    run_on_omp();
}