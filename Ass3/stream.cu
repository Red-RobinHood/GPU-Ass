#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace std;

string OUTPUT_FILE;

#define threads 256

__global__ void computeNextQueue(const int *currqueue, int currsize,
                                 int *distance, int *visited, int *nextsize,
                                 int *nextqueue, int level, float *currvectors,
                                 int dim, float *startvector,
                                 float *diststart) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < currsize; idx += stride) {
    int v = currqueue[idx];
    int old = atomicExch(&visited[v], 1);
    if (old == 0) {
      distance[v] = level;
      int pos = atomicAdd(nextsize, 1);
      nextqueue[pos] = v;
      float dist = 0.0f;
      for (int d = 0; d < dim; d++) {
        float diff = startvector[d] - currvectors[idx * dim + d];
        dist += diff * diff;
      }
      diststart[v] = sqrtf(dist);
    }
  }
}

vector<vector<float>> read_fvecs(const string &filename) {
  ifstream f(filename, ios::binary);
  vector<vector<float>> data;
  while (true) {
    int dim;
    f.read(reinterpret_cast<char *>(&dim), 4);
    if (!f)
      break;
    vector<float> v(dim);
    f.read(reinterpret_cast<char *>(v.data()), dim * sizeof(float));
    data.push_back(std::move(v));
  }
  return data;
}

float caldist(vector<float> a, vector<float> b) {
  float dist = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return sqrt(dist);
}

void printVerticesByLevel(const vector<int> &distance, int start,
                          const string &dataset_path,
                          vector<float> diststart_host, int duration) {
  vector<vector<float>> vecs = read_fvecs(dataset_path);
  unordered_map<int, vector<int>> levelMap;

  for (int i = 0; i < (int)distance.size(); i++) {
    levelMap[distance[i]].push_back(i);
  }

  ofstream fout(OUTPUT_FILE);

  for (int level = 0; levelMap.find(level) != levelMap.end(); level++) {
    int c = -1, f = -1, count = 0;
    float d_c = 100000000, d_f = -1;
    for (int v : levelMap[level]) {
      count++;
      float dist = diststart_host[v];
      if (dist < d_c) {
        c = v;
        d_c = dist;
      }
      if (dist > d_f) {
        f = v;
        d_f = dist;
      }
    }
    cout << level << "," << count << "," << f << "," << d_f << "," << c << ","
         << d_c << "\n";
    fout << level << "," << count << "," << f << "," << d_f << "," << c << ","
         << d_c << "\n";
  }
  cout << "Total BFS discovery time = " << duration << " ms";
  fout << "Total BFS discovery time = " << duration << " ms";
  fout.close();
}

void bfsGPU_streams(int start, const vector<vector<int>> &graph,
                    const string &dataset_path) {
  int n = graph.size();
  int neighbours = graph[0].size();
  vector<vector<float>> vecs = read_fvecs(dataset_path);
  int dim = vecs[0].size();

  size_t maxChunk = (n * neighbours + 3) / 4;
  size_t currQueueBytes = maxChunk * sizeof(int);
  size_t nextQueueBytes = maxChunk * neighbours * sizeof(int);
  size_t currVectorsBytes = maxChunk * dim * sizeof(float);

  int *d_currqueue[4], *d_nextqueue[4], *d_nextsize[4];
  int *d_distance, *d_visited;
  float *d_currvectors[4], *d_startvector, *d_diststart;
  int currsize[4] = {1, 1, 1, 1};
  cudaStream_t streams[4];

  for (int i = 0; i < 4; ++i) {
    cudaMalloc(&d_currqueue[i], currQueueBytes);
    cudaMalloc(&d_nextqueue[i], nextQueueBytes);
    cudaMalloc(&d_nextsize[i], sizeof(int));
    cudaMalloc(&d_currvectors[i], currVectorsBytes);
    cudaStreamCreate(&streams[i]);
  }
  cudaMalloc(&d_distance, n * sizeof(int));
  cudaMalloc(&d_visited, n * sizeof(int));
  cudaMalloc(&d_startvector, dim * sizeof(float));
  cudaMalloc(&d_diststart, n * sizeof(float));

  vector<int> h_distance(n, -1);
  vector<int> h_visited(n, 0);
  vector<float> h_diststart(n, 0.0f);
  h_distance[start] = 0;
  h_visited[start] = 1;
  cudaMemcpy(d_distance, h_distance.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_visited, h_visited.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_startvector, vecs[start].data(), dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_diststart, h_diststart.data(), n * sizeof(float),
             cudaMemcpyHostToDevice);

  int level = 1;
  const int zero = 0;
  int startBase = neighbours / 4;
  int startRem = neighbours % 4;
  bool cond;
  auto t0 = chrono::steady_clock::now();

  for (int i = 0; i < 4; ++i) {
    int chunk_start = i * startBase;
    int chunk_size = startBase;
    if (i == 3)
      chunk_size += startRem;
    vector<float> currvectors_host(chunk_size * dim);
    for (int j = 0; j < chunk_size; j++) {
      int node_id = graph[start][chunk_start + j];
      for (int d = 0; d < dim; d++) {
        currvectors_host[j * dim + d] = vecs[node_id][d];
      }
    }

    cudaMemcpyAsync(d_nextsize[i], &zero, sizeof(int), cudaMemcpyHostToDevice,
                    streams[i]);
    cudaMemcpyAsync(d_currqueue[i], graph[start].data() + chunk_start,
                    chunk_size * sizeof(int), cudaMemcpyHostToDevice,
                    streams[i]);
    cudaMemcpyAsync(d_currvectors[i], currvectors_host.data(),
                    chunk_size * dim * sizeof(float), cudaMemcpyHostToDevice,
                    streams[i]);
    computeNextQueue<<<1, threads, 0, streams[i]>>>(
        d_currqueue[i], chunk_size, d_distance, d_visited, d_nextsize[i],
        d_nextqueue[i], level, d_currvectors[i], dim, d_startvector,
        d_diststart);
  }
  do {
    level++;

    vector<int> next;
    vector<vector<int>> temp(4);
    int oldsize[4], totalSize = 0;

    for (int i = 0; i < 4; i++) {
      oldsize[i] = currsize[i];
      temp[i].resize(oldsize[i] * neighbours);
      cudaMemcpyAsync(&currsize[i], d_nextsize[i], sizeof(int),
                      cudaMemcpyDeviceToHost, streams[i]);
      cudaMemcpyAsync(temp[i].data(), d_nextqueue[i],
                      oldsize[i] * neighbours * sizeof(int),
                      cudaMemcpyDeviceToHost, streams[i]);
    }

    for (int i = 0; i < 4; i++) {
      cudaStreamSynchronize(streams[i]);
      currsize[i] = min(currsize[i], oldsize[i] * neighbours);
      totalSize += currsize[i];
    }

    next.reserve(totalSize);
    for (int i = 0; i < 4; i++) {
      next.insert(next.end(), temp[i].begin(), temp[i].begin() + currsize[i]);
    }

    vector<int> nextlist;
    vector<float> nextvectors;
    nextlist.reserve(totalSize * neighbours);
    nextvectors.reserve(totalSize * neighbours * dim);
    for (int v : next) {
      for (int j = 0; j < neighbours; j++) {
        int neighbor_id = graph[v][j];
        nextlist.push_back(neighbor_id);
        for (int d = 0; d < dim; d++) {
          nextvectors.push_back(vecs[neighbor_id][d]);
        }
      }
    }

    int total = nextlist.size();
    int base = total / 4;
    int rem = total % 4;

    for (int i = 0; i < 4; i++) {
      int chunk_start = i * base;
      int chunk_size = base;
      if (i == 3)
        chunk_size += rem;

      cudaMemcpyAsync(d_nextsize[i], &zero, sizeof(int), cudaMemcpyHostToDevice,
                      streams[i]);
      cudaMemcpyAsync(d_currqueue[i], nextlist.data() + chunk_start,
                      chunk_size * sizeof(int), cudaMemcpyHostToDevice,
                      streams[i]);
      cudaMemcpyAsync(d_currvectors[i], nextvectors.data() + chunk_start * dim,
                      chunk_size * dim * sizeof(float), cudaMemcpyHostToDevice,
                      streams[i]);
      computeNextQueue<<<1, threads, 0, streams[i]>>>(
          d_currqueue[i], chunk_size, d_distance, d_visited, d_nextsize[i],
          d_nextqueue[i], level, d_currvectors[i], dim, d_startvector,
          d_diststart);
    }

    cond = currsize[0] || currsize[1] || currsize[2] || currsize[3];

  } while (cond);

  cudaMemcpy(h_distance.data(), d_distance, n * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_diststart.data(), d_diststart, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  auto t1 = chrono::steady_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
  printVerticesByLevel(h_distance, start, dataset_path, h_diststart, duration);

  for (int i = 0; i < 4; i++) {
    cudaFree(d_currqueue[i]);
    cudaFree(d_nextqueue[i]);
    cudaFree(d_nextsize[i]);
    cudaFree(d_currvectors[i]);
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(d_distance);
  cudaFree(d_visited);
  cudaFree(d_startvector);
  cudaFree(d_diststart);
}

vector<vector<int>> readCSVtoAdjList(const string &filename) {
  vector<vector<int>> graph;
  ifstream file(filename);

  string line;
  if (!file.eof())
    getline(file, line);

  while (getline(file, line)) {
    stringstream ss(line);
    string value;
    vector<int> nums;

    while (getline(ss, value, ',')) {
      if (!value.empty())
        nums.push_back(stoi(value));
    }
    int idx = nums[0];
    if ((int)graph.size() <= idx)
      graph.resize(idx + 1);

    graph[idx] = vector<int>(nums.begin() + 1, nums.end());
  }

  file.close();
  return graph;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cout << "Usage:\n";
    cout << "./a.out <ABS_PATH_GRAPH_CSV> <ABS_PATH_DATASET_FVECS> "
            "<NODE_ID>\n";
    return 1;
  }

  string graph_path = argv[1];
  string dataset_path = argv[2];
  int start_node = stoi(argv[3]);

  string graph_filename = graph_path.substr(graph_path.find_last_of("/") + 1);
  graph_filename = graph_filename.substr(0, graph_filename.size() - 4);
  OUTPUT_FILE = "output_" + graph_filename + "_task_2.txt";

  vector<vector<int>> graph = readCSVtoAdjList(graph_path);

  bfsGPU_streams(start_node, graph, dataset_path);

  return 0;
}