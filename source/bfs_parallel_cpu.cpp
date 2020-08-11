//
// Converted from Grant's implementation in graph.c on 8/2/20.
//

#include "bfs_parallel_cpu.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
  // This implementation is based on the implementation found here:
  // https://github.com/progschj/ThreadPool
  // so...
  // Copyright (c) 2012 Jakob Progsch, Václav Zeman
  //
  // This software is provided 'as-is', without any express or implied
  // warranty. In no event will the authors be held liable for any damages
  // arising from the use of this software.
  //
  // Permission is granted to anyone to use this software for any purpose,
  // including commercial applications, and to alter it and redistribute it
  // freely, subject to the following restrictions:
  //
  //    1. The origin of this software must not be misrepresented; you must not
  //    claim that you wrote the original software. If you use this software
  //    in a product, an acknowledgment in the product documentation would be
  //    appreciated but is not required.
  //
  //    2. Altered source versions must be plainly marked as such, and must not be
  //    misrepresented as being the original software.
  //
  //    3. This notice may not be removed or altered from any source
  //    distribution.

 public:
  ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}
  explicit ThreadPool(unsigned int num_threads) : stopped(false) {
    for (std::size_t i = 0; i < num_threads; i++) {
      workers.emplace_back([this](){
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(lock, [this]()->bool{
              return this->stopped or not this->task_queue.empty();
            });
            if (this->stopped and this->task_queue.empty()) {
              return;
            }
            task = std::move(this->task_queue.front());
            this->task_queue.pop();
          }
          task();
        }
      });
    }
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stopped = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }

  template <typename Function, typename ... Args>
  std::future<std::invoke_result_t<Function, Args...>> Add(Function function, Args&& ... args) {
    auto task = std::make_shared<std::packaged_task<std::invoke_result_t<Function, Args...>()>>(
        std::bind(std::forward<Function>(function), std::forward<Args>(args)...));
    std::future<std::invoke_result_t<Function, Args...>> result = task->get_future();
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      task_queue.emplace([task](){(*task)();});
    }
    condition.notify_one();
    return result;
  }

 protected:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> task_queue;
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stopped;
};

template <int unreached>
struct BFS_Worker {
  BFS_Worker(ThreadPool& _pool, const Graph& _graph, std::atomic<int>* _distances)
      : pool(_pool), graph(_graph), distances(_distances), current_distance(unreached) {};
  ThreadPool& pool;
  const Graph& graph;
  std::atomic<int>* distances;
  int current_distance;

  void operator()(Graph::index_type parent) {
    int parent_distance = distances[parent].load();
    auto iterator = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(parent);
    auto end = graph.matrix.column_indices.begin() + graph.matrix.row_indices.at(parent + 1);
    for (; iterator < end; iterator++) {
      if (distances[*iterator].compare_exchange_strong(current_distance, parent_distance + 1)) {
        pool.Add(BFS_Worker<unreached>(pool, graph, distances), current_distance);
      }
    }
  }
};

std::vector<int> BFS_ParallelCPU(const Graph& graph, Graph::index_type source) {
  static constexpr int unreached = -1;
  auto* atomic_distances = new std::atomic<int>[graph.NumNodes()];
  std::for_each(atomic_distances, atomic_distances + graph.NumNodes(),
      [](std::atomic<int>& val){
    val.store(unreached);
  });
  atomic_distances[source].store(0);

  {
    ThreadPool pool;
    pool.Add(BFS_Worker<unreached>{pool, graph, atomic_distances}, source);
  } // Exit scope to join threads

  std::vector<int> distances(graph.NumNodes());
  std::transform(atomic_distances, atomic_distances + graph.NumNodes(), distances.begin(),
      [](std::atomic<int>& val)->int{
    return val.load();
  });
  delete[] atomic_distances;
  return distances;
}