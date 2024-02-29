#include <omp.h>

#include <iostream>

 

constexpr int N = 4;

int main() {

int v[N];

for (int i = 0; i < N; i++) v[i] = i;

 

#pragma omp target enter data map(to: v)  

#pragma omp target teams distribute parallel for simd

for (int i = 0; i < N; i++) {

   v[i] += 4 + i;

}

 

for (int i = 0; i < N; i++) std::cout << v[i] << " ";

 

#pragma omp target teams distribute parallel for simd

for (int i = 0; i < N; i++) {

   v[i] *= v[i] * i;

}

 

#pragma omp target exit data map(from: v)

for (int i = 0; i < N; i++) std::cout << v[i] << " ";

 return 0;

}