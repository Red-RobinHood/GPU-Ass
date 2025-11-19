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

__global__ void computeNextQueue(int *currqueue, int currsize, int *distance,
                                 int *visited, int *nextsize, int *nextqueue,
                                 int level, float *curr_vectors, int dim,
                                 float *start_vector, float *diststart) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= currsize)
    return;
  int v = currqueue[tid];
  int old = atomicExch(&visited[v], 1);
  if (!old) {
    distance[v] = level;
    int position = atomicAdd(nextsize, 1);
    nextqueue[position] = v;
    float dist = 0.0f;
    for (int d = 0; d < dim; d++) {
      float diff = start_vector[d] - curr_vectors[tid * dim + d];
      dist += diff * diff;
    }
    diststart[v] = sqrtf(dist);
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

void bfsGPU(int start, vector<vector<int>> graph, const string &dataset_path) {
  int neighbours = graph[0].size();
  int n = graph.size();
  vector<vector<float>> vecs = read_fvecs(dataset_path);
  int dim = vecs[0].size();

  int *currqueue;
  int *distance;
  int *visited;
  int *nextsize;
  int *nextqueue;
  float *currvectors;
  float *startvector;
  float *diststart;
  int nextqueue_host[n];

  int level = 1;
  int currsize = 1;

  cudaMalloc((void **)&currqueue, n * neighbours * sizeof(int));
  cudaMalloc((void **)&distance, n * sizeof(int));
  cudaMalloc((void **)&visited, n * sizeof(int));
  cudaMalloc((void **)&nextsize, sizeof(int));
  cudaMalloc((void **)&nextqueue, n * sizeof(int));
  cudaMalloc((void **)&currvectors, n * neighbours * dim * sizeof(float));
  cudaMalloc((void **)&startvector, dim * sizeof(float));
  cudaMalloc((void **)&diststart, n * sizeof(float));

  vector<int> distance_host(n, -1);
  distance_host[start] = 0;
  vector<int> visited_host(n, 0);
  visited_host[start] = true;
  vector<float> diststart_host(n, 0.0f);

  auto startTime = chrono::steady_clock::now();

  cudaMemcpy(distance, distance_host.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(visited, visited_host.data(), n * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(startvector, vecs[start].data(), dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(diststart, diststart_host.data(), n * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(nextsize, 0, sizeof(int));

  vector<float> currvectors_host(neighbours * dim);
  for (int i = 0; i < neighbours; i++) {
    int node_id = graph[start][i];
    for (int d = 0; d < dim; d++) {
      currvectors_host[i * dim + d] = vecs[node_id][d];
    }
  }
  cudaMemcpy(currqueue, graph[start].data(), neighbours * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(currvectors, currvectors_host.data(),
             neighbours * dim * sizeof(float), cudaMemcpyHostToDevice);

  currsize = neighbours;

  while (currsize > 0) {
    computeNextQueue<<<currsize / neighbours, neighbours>>>(
        currqueue, currsize, distance, visited, nextsize, nextqueue, level,
        currvectors, dim, startvector, diststart);
    cudaDeviceSynchronize();
    level++;
    cudaMemcpy(&currsize, nextsize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(nextqueue_host, nextqueue, currsize * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemset(nextsize, 0, sizeof(int));

    vector<int> nextlist;
    vector<float> nextvectors;
    for (int i = 0; i < currsize; i++) {
      int node = nextqueue_host[i];
      for (int j = 0; j < neighbours; j++) {
        int neighbor_id = graph[node][j];
        nextlist.push_back(neighbor_id);
        for (int d = 0; d < dim; d++) {
          nextvectors.push_back(vecs[neighbor_id][d]);
        }
      }
    }

    currsize = nextlist.size();
    if (currsize > 0) {
      cudaMemcpy(currqueue, nextlist.data(), currsize * sizeof(int),
                 cudaMemcpyHostToDevice);
      cudaMemcpy(currvectors, nextvectors.data(),
                 currsize * dim * sizeof(float), cudaMemcpyHostToDevice);
    }
  }

  cudaMemcpy(distance_host.data(), distance, n * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(diststart_host.data(), diststart, n * sizeof(float),
             cudaMemcpyDeviceToHost);
  auto endTime = chrono::steady_clock::now();
  auto duration =
      chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
  printVerticesByLevel(distance_host, start, dataset_path, diststart_host,
                       duration);
  cudaFree(currqueue);
  cudaFree(distance);
  cudaFree(visited);
  cudaFree(nextsize);
  cudaFree(nextqueue);
  cudaFree(currvectors);
  cudaFree(startvector);
  cudaFree(diststart);
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
  OUTPUT_FILE = "output_" + graph_filename + "_task_1.txt";

  vector<vector<int>> graph = readCSVtoAdjList(graph_path);

  bfsGPU(start_node, graph, dataset_path);

  return 0;
}