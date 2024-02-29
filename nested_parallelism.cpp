#include <iostream>
#include <limits>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

double rand_uniform();

int main()
{
    try {
        int m = 600;
        int k = 1200;
        int n = 128;

        // Create a queue on the default device.
        sycl::queue device_queue{sycl::default_selector{}};

        std::cout << "Device: "
                  << device_queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // Allocate shared memory for matrices.
        auto A = sycl::malloc_shared<double>(m * k, device_queue);
        auto B = sycl::malloc_shared<double>(k * n, device_queue);
        auto C = sycl::malloc_shared<double>(m * n, device_queue);

        // Initialize matrix data.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                A[i * k + j] = rand_uniform();

        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                B[i * n + j] = rand_uniform();

        std::cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";

        sycl::range<2> num_groups{(size_t)m, (size_t)k};
        sycl::range<2> group_size{(size_t)n, 1};

        device_queue.submit([&](sycl::handler &h) {
            h.parallel_for_work_group(num_groups, [=](sycl::group<2> grp) {
                grp.parallel_for_work_item(group_size, [&](sycl::h_item<2> item) {
                    auto idx = grp.get_id(0) * item.get_local_id(0) + grp.get_id(1);
                    C[idx] = A[idx] + B[idx];
                });    
            });
        });


        // Wait for oneMKL computation to complete.
        device_queue.wait_and_throw();

        // Free memory.
        free(A, device_queue);
        free(B, device_queue);
        free(C, device_queue);

    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);
    }
}

double rand_uniform()
{
    return double(rand()) / RAND_MAX;
}
