#!/bin/bash

nvcc main_1_1.cu -o matmul_1d
nvcc main_1_2.cu -o matmul_2d
nvcc main_2.cu -o matmul_tiled
nvcc main_3.cu -o mattrans_basic
nvcc main_4.cu -o mattrans_tiled