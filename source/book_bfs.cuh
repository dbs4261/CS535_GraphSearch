//
// Created by Daniel Simon on 7/26/20.
//

#ifndef CS535_GRAPHSEARCH_BOOK_BFS_CUH
#define CS535_GRAPHSEARCH_BOOK_BFS_CUH

//TODO: placeholder values
#ifndef BLOCK_QUEUE_SIZE
#define BLOCK_QUEUE_SIZE (32 * 128)
#endif

namespace book {

__global__ void BFS_Bqueue_kernel(unsigned int* p_frontier, unsigned int* p_frontier_tail, unsigned int* c_frontier,
    unsigned int* c_frontier_tail, unsigned int* edges, unsigned int* dest, unsigned int* label, unsigned int* visited);

cudaError_t Launch_BFS_Bqueue_kernel(dim3 grid_size, dim3 block_size, unsigned int* p_frontier,
    unsigned int* p_frontier_tail, unsigned int* c_frontier, unsigned int* c_frontier_tail,
    unsigned int* edges, unsigned int* dest, unsigned int* label, unsigned int* visited);

}

#endif //CS535_GRAPHSEARCH_BOOK_BFS_CUH
