add_executable(Main main.cpp)
target_link_libraries(Main
    PUBLIC
        BFSLibrary
        ${CUDA_LIBRARIES}
)

set_source_files_properties(regular.c PROPERTIES LANGUAGE C)
add_executable(CudaRegularBFS regular.c)
target_link_libraries(CudaRegularBFS
    PUBLIC
        BFSLibrary
        ${CUDA_LIBRARIES}
)