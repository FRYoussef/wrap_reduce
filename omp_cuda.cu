cudaStream_t stream;
cublasHandle_t handle_gemm1;
cublasCreate(&handle_gemm1);


#pragma omp target data \
    map(to:image_vector[0:linessamplesbands]) \
    map(to:endmembers[0:targets*bands])\
    map(from:abundanceVector[0:lines_samples*targets]) \
    map(alloc:h_Num[0:lines_samples*targets])\
    map(alloc:h_aux[0:lines_samples*bands])\
    map(alloc:h_Den[0:lines_samples*targets])
{


    #pragma omp target teams distribute parallel for
    for(i=0; i<lines_samples*targets; i++)
        abundanceVector[i]=1;
    
 
    double alpha = 1.0, beta = 0.0;


    #pragma omp target data use_device_ptr(image_vector,endmembers,h_Num)
    {
        int cublas_error = cublasDgemm(handle_gemm1,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                lines_samples, targets, bands,
                                &alpha, 
                                image_vector, lines_samples, 
                                endmembers, targets, 
                                &beta, 
                                h_Num, lines_samples);
    }