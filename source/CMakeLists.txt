add_library(CudaLibrary
    ${CMAKE_CURRENT_LIST_DIR}/library.cu
    ${CMAKE_CURRENT_LIST_DIR}/library.cuh
    ${CMAKE_CURRENT_LIST_DIR}/error_checker.h
    ${CMAKE_CURRENT_LIST_DIR}/book_bfs.cu
    ${CMAKE_CURRENT_LIST_DIR}/book_bfs.cuh
    ${CMAKE_CURRENT_LIST_DIR}/bfs_sequential.hpp
    ${CMAKE_CURRENT_LIST_DIR}/bfs_sequential.cpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/sparse_matrix.hpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/sparse_matrix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/graph.hpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/graph.cpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/graph_utilities.hpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/graph_utilities.cpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/coloring.hpp
    ${CMAKE_CURRENT_LIST_DIR}/graphs/coloring.cpp
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

if(OpenACC_CXX_FOUND)
    message("Adding OpanACC support")
    target_compile_definitions(CudaLibrary PUBLIC ENABLE_OPENACC)
    target_sources(CudaLibrary
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_openacc.hpp
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_openacc.cpp)
    set_source_files_properties(bfs_openacc.cpp PROPERTIES COMPILE_FLAGS ${OpenACC_CXX_FLAGS})
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    message("Adding gpu parallel STL implementation")
    target_compile_definitions(CudaLibrary PUBLIC ENABLE_PARALLEL_STL)
    target_sources(CudaLibrary
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_parallel_stl.hpp
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_parallel_stl.cpp
    )
    set_source_files_properties(bfs_parallel_stl.cpp PROPERTIES COMPILE_FLAGS -stdpar)
endif()