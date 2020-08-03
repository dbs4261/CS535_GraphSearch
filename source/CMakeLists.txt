add_library(CudaLibrary
    library.cu
    library.cuh
    error_checker.h
    book_bfs.cu
    book_bfs.cuh
    sparse_matrix.hpp
    graph.hpp
    graph_utilities.hpp
    bfs_sequential.hpp
)

target_include_directories(CudaLibrary
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
    INTERFACE
        ${CUDA_INCLUDE_DIRS}
)

target_compile_options(CudaLibrary
    PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_COMPILED_ARCH}>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
        $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-g>>
        $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-G>>
        $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:-O0>>
        $<$<CONFIG:DEBUG>:$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>>
)