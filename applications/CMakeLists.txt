add_executable(Main main.cpp)
target_link_libraries(Main
    PUBLIC
        CudaLibrary
        ${CUDA_LIBRARIES}
)