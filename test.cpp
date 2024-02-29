// reduce previous peeled for loop
for (size_t i = 1; i < B; i++)
    result[0] += result[i];

// reduce simd lane into scalar var
for(int i{0}; i < simd_width; i++)
    distance += result[0][i];

// calculate remaining values
for(int d{dims - simd_remainder}; d < dims; d++)
    distance += squared_l2_distance(points[(i * dims) + d], means[(cluster * dims) + d]);