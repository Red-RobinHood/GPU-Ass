#!/bin/bash
nvcc -arch=sm_80 baseline.cu -o bfs_baseline
nvcc -arch=sm_80 stream.cu -o bfs_stream
nvcc -arch=sm_80 graph.cu -o bfs_graph