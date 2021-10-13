
#include "GraphPieces.h"

#include <cuda_runtime.h>

#include <queue>
#include <set>
#include <vector>

#include "CUDASleep.h"
#include "utils.h"

static cudaGraphExec_t GraphSubset(std::set<cudaGraphNode_t> lnodes,
                                   cudaGraph_t graph) {
  std::vector<cudaGraphNode_t> nodes;
  size_t nr;
  CUDA_API_CALL(cudaGraphGetNodes(graph, nullptr, &nr));
  nodes.resize(nr);
  CUDA_API_CALL(cudaGraphGetNodes(graph, nodes.data(), &nr));

  cudaGraph_t gclone;
  CUDA_API_CALL(cudaGraphClone(&gclone, graph));
  for (auto& n : nodes) {
    if (lnodes.count(n)) continue;
    cudaGraphNode_t clnode;
    CUDA_API_CALL(cudaGraphNodeFindInClone(&clnode, n, gclone));
    CUDA_API_CALL(cudaGraphDestroyNode(clnode));
  }

  cudaGraphExec_t exec;
  CUDA_API_CALL(cudaGraphInstantiate(&exec, gclone, nullptr, nullptr, 0));
  CUDA_API_CALL(cudaGraphDestroy(gclone));

  return exec;
}

static float TimeGraphNode(std::set<cudaGraphNode_t> lnodes,
                           cudaGraph_t graph) {
  cudaGraphExec_t exec = GraphSubset(lnodes, graph);
  CUDA_API_CALL(cudaGraphUpload(exec, 0));

  cudaEvent_t begin, end;
  CUDA_API_CALL(cudaEventCreateWithFlags(&begin, cudaEventDefault));
  CUDA_API_CALL(cudaEventCreateWithFlags(&end, cudaEventDefault));
  CUDA_API_CALL(cudaDeviceSynchronize());
  gpu_nsleep(5000000, 0);
  CUDA_API_CALL(cudaEventRecord(begin));
  CUDA_API_CALL(cudaGraphLaunch(exec, 0));
  CUDA_API_CALL(cudaEventRecord(end));
  CUDA_API_CALL(cudaDeviceSynchronize());

  float ms;
  CUDA_API_CALL(cudaEventElapsedTime(&ms, begin, end));
  CUDA_API_CALL(cudaEventDestroy(begin));
  CUDA_API_CALL(cudaEventDestroy(end));

  CUDA_API_CALL(cudaGraphExecDestroy(exec));

  return ms;
}

GraphPieces::~GraphPieces() {
  for (auto& p : parts_) CUDA_API_CALL(cudaGraphExecDestroy(p));
}

std::shared_ptr<GraphPieces> GraphPieces::MergePieces(
    std::vector<std::shared_ptr<GraphPieces>> pieces) {
  std::vector<cudaGraphExec_t> excs;

  for (auto& p : pieces) {
    excs.insert(excs.end(), p->parts_.begin(), p->parts_.end());
    p->parts_.clear();
  }

  return std::shared_ptr<GraphPieces>(new GraphPieces(excs));
}

std::shared_ptr<GraphPieces> GraphPieces::GraphsToSingleExec(
    std::vector<cudaGraph_t> parts, size_t duplicatorcount) {
  cudaGraph_t newGraph;
  cudaGraphNode_t curDep;

  size_t nparts_done = 0;

  CUDA_API_CALL(cudaGraphCreate(&newGraph, 0));

  for (size_t i = 0; i < duplicatorcount; i++) {
    for (auto& p : parts) {
      CUDA_API_CALL(cudaGraphAddChildGraphNode(&curDep, newGraph,
                                               nparts_done ? &curDep : nullptr,
                                               nparts_done ? 1 : 0, p));
      nparts_done++;
    }
  }

  cudaGraphExec_t exc;
  CUDA_API_CALL(cudaGraphInstantiate(&exc, newGraph, nullptr, nullptr, 0));
  CUDA_API_CALL(cudaGraphDestroy(newGraph));
  return std::shared_ptr<GraphPieces>(new GraphPieces({exc}));
}

std::shared_ptr<GraphPieces> GraphPieces::GraphToExecs(cudaGraph_t graph,
                                                       float ms_piece_split) {
  assert(graph != nullptr);

  if (ms_piece_split <= 0) {
    cudaGraph_t clone;
    CUDA_API_CALL(cudaGraphClone(&clone, graph));
    cudaGraphExec_t gr;
    CUDA_API_CALL(cudaGraphInstantiate(&gr, clone, nullptr, nullptr, 0));
    CUDA_API_CALL(cudaGraphDestroy(clone));
    return std::shared_ptr<GraphPieces>(new GraphPieces({gr}));
  }

  size_t nr;

  std::vector<cudaGraphNode_t> nodes;
  std::queue<cudaGraphNode_t> stack;
  std::set<cudaGraphNode_t> seen;

  assert(graph != nullptr);
  CUDA_API_CALL(cudaGraphGetRootNodes(graph, nullptr, &nr));
  nodes.resize(nr);
  CUDA_API_CALL(cudaGraphGetRootNodes(graph, nodes.data(), &nr));
  // assert(nr == 1);

  std::vector<std::set<cudaGraphNode_t>> layers;
  std::vector<float> layers_time;

  std::set<cudaGraphNode_t> cur_layer;

  auto visit = [&](std::vector<cudaGraphNode_t>& nodes) {
    for (auto& p : nodes) {
      if (seen.count(p) != 0) continue;
      seen.insert(p);
      stack.push(p);
      cur_layer.insert(p);
    }
  };

  visit(nodes);
  float ms_tot = 0.0;

  while (stack.size() > 0) {
    while (stack.size() > 1) {
      auto node = stack.front();
      stack.pop();

      CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nullptr, &nr));
      if (!nr) continue;
      nodes.resize(nr);
      CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nodes.data(), &nr));
      visit(nodes);
    }

    if (stack.size() == 0)  // Handle?
      continue;

    assert(stack.size() == 1);
    layers.push_back(cur_layer);

    float layer_ms = TimeGraphNode(cur_layer, graph);
    layers_time.push_back(layer_ms);
    ms_tot += layer_ms;
    cur_layer.clear();

    auto node = stack.front();
    stack.pop();

    CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nullptr, &nr));
    if (!nr) continue;
    nodes.resize(nr);
    CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nodes.data(), &nr));
    visit(nodes);
  }

  assert(stack.size() == 0);
  assert(cur_layer.size() == 0);

  float cur_layer_ms = 0;

  cur_layer.clear();
  std::vector<cudaGraphExec_t> merged_layers;

  for (size_t i = 0; i < layers.size(); i++) {
    auto& l = layers.at(i);
    cur_layer.insert(l.begin(), l.end());
    cur_layer_ms += layers_time.at(i);

    if (cur_layer_ms >= ms_piece_split) {
      merged_layers.push_back(GraphSubset(cur_layer, graph));
      cur_layer.clear();
      cur_layer_ms = 0;
    }
  }
  merged_layers.push_back(GraphSubset(cur_layer, graph));

  return std::shared_ptr<GraphPieces>(new GraphPieces(merged_layers));
}
