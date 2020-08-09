add_executable(Main main.cpp)
target_link_libraries(Main
    PUBLIC
        BFSLibrary
        ${CUDA_LIBRARIES}
)

set_source_files_properties(graph.c PROPERTIES LANGUAGE C)
add_executable(SequentialBFS graph.c)