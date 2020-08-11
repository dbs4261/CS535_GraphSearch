add_library(BFSLibrary
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
    ${CMAKE_CURRENT_LIST_DIR}/cuda_regular_bfs.cu
    ${CMAKE_CURRENT_LIST_DIR}/cuda_regular_bfs.h
    ${CMAKE_CURRENT_LIST_DIR}/graph.c
    ${CMAKE_CURRENT_LIST_DIR}/graph.h
    ${CMAKE_CURRENT_LIST_DIR}/allocate_for_cuda_bfs.c
    ${CMAKE_CURRENT_LIST_DIR}/allocate_for_cuda_bfs.h
    ${CMAKE_CURRENT_LIST_DIR}/timing.hpp
    ${CMAKE_CURRENT_LIST_DIR}/warp_queue_bfs.cu
    ${CMAKE_CURRENT_LIST_DIR}/warp_queue_bfs.h
)

target_include_directories(BFSLibrary
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../>
        ${CUDA_INCLUDE_DIRS}
)

target_compile_options(BFSLibrary
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
    target_compile_definitions(BFSLibrary PUBLIC ENABLE_OPENACC)
    target_sources(BFSLibrary
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_openacc.hpp
        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_openacc.cpp)
    set_source_files_properties(bfs_openacc.cpp PROPERTIES COMPILE_FLAGS ${OpenACC_CXX_FLAGS})
endif()

#if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
#    message("Adding gpu parallel STL implementation")
#    target_compile_definitions(BFSLibrary PUBLIC ENABLE_PARALLEL_STL)
#    target_sources(BFSLibrary
#        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_parallel_stl.hpp
#        PUBLIC ${CMAKE_CURRENT_LIST_DIR}/bfs_parallel_stl.cpp
#    )
#    set_source_files_properties(bfs_parallel_stl.cpp PROPERTIES COMPILE_FLAGS -stdpar)
#endif()

add_executable(YubinBFS origin_bfs.cu allocate_for_cuda_bfs.c)
target_include_directories(YubinBFS
    PUBLIC
        ${CUDA_INCLUDE_DIRS}
)
target_compile_options(YubinBFS
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