#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <ctime>
#include "include/structs.h"
#include "include/kNN_gpu.h"

using namespace std;

__device__
bool checkIfInsideSubspace(Node& node, Point& query, unsigned int dim) {
    for(int d = 0; d < dim; d++) {
        if(query.gpu[d] < node.startPoint.gpu[d] || query.gpu[d] > node.endPoint.gpu[d]) return false;
    }
    return true;
}

__global__
void runnable(Node* buffer, Point* queries, int* queryIds, int* traverseRoot, Point* nearest, int qsize, int dim)
{
    // designate query point to be process by this thread
    int q = threadIdx.x + blockIdx.x * blockDim.x;
    // do nothing if query exceeds total number of queries
    if(q >= qsize) return;
    long min_dist = __LONG_MAX__;
    long curr_dist;
    unsigned int currentOffs = traverseRoot[q];
    Stack nodeStack;
    push(nodeStack, currentOffs);
    // keep examining tree until stack is empty
    while(!isEmpty(nodeStack)) {
        // pop offset of node to visit from buffer
        currentOffs = pop(nodeStack);
        if(buffer[currentOffs].points.len > 0) {
            curr_dist = 0;
            for(int d = 0; d < dim; d++) {
                curr_dist += (queries[q].gpu[d] - buffer[currentOffs].points.gpu[d]) * (queries[q].gpu[d] - buffer[currentOffs].points.gpu[d]);
            }
            if(curr_dist < min_dist) {
                min_dist = curr_dist;
                for(int d = 0; d < dim; d++) nearest[queryIds[q]].gpu[d] = buffer[currentOffs].points.gpu[d];
            }
        }
        // if non-leaf node, keep traversing
        if(buffer[currentOffs].left.gpu != -1 && buffer[currentOffs].right.gpu != -1) {
            if(buffer[currentOffs].nodeType == METRIC_NODE) {
                push(nodeStack, buffer[currentOffs].right.gpu);
                push(nodeStack, buffer[currentOffs].left.gpu);
            }
            else if (buffer[currentOffs].nodeType == SPILL_NODE) {
                if(checkIfInsideSubspace(buffer[buffer[currentOffs].right.gpu], queries[q], dim)) push(nodeStack, buffer[currentOffs].right.gpu);
                if(checkIfInsideSubspace(buffer[buffer[currentOffs].left.gpu], queries[q], dim)) push(nodeStack, buffer[currentOffs].left.gpu);
            }
        }
    }
}

__host__
void kNN_gpu(Buffer buffer, GroupPoint* groupPoints, vector<Point>& cpu_nearest, int dim, int numQueries, int numGroups) {
    Point* gpu_queries;
    int* queryIds;
    int* traverseRoot;
    Point* gpu_nearest;

    // copy data points
    for(int i = 0; i < buffer.pointer; i++) {
        buffer.get(i)->points.len = buffer.get(i)->points.cpu.size();
        // copy point info, starting and ending points into gpu
        if(buffer.get(i)->points.len > 0) cudaMallocManaged(&buffer.get(i)->points.gpu, dim);
        cudaMallocManaged(&buffer.get(i)->startPoint.gpu, dim * sizeof(float));
        cudaMallocManaged(&buffer.get(i)->endPoint.gpu, dim * sizeof(float));
        for(int d = 0; d < dim; d++) {
            if(buffer.get(i)->points.len > 0) buffer.get(i)->points.gpu[d] = buffer.get(i)->points.cpu.at(0).cpu.at(d);
            buffer.get(i)->startPoint.gpu[d] = buffer.get(i)->startPoint.cpu.at(d);
            buffer.get(i)->endPoint.gpu[d] = buffer.get(i)->endPoint.cpu.at(d);
        }
    }

    // copy query points
    cudaMallocManaged(&gpu_queries, numQueries * sizeof(*gpu_queries));
    cudaMallocManaged(&queryIds, numQueries * sizeof(*queryIds));
    cudaMallocManaged(&traverseRoot, numQueries * sizeof(*traverseRoot));
    int q = 0;
    for(int g = 0; g < numGroups; g++) {
        for(int p = 0; p < groupPoints[g].memberPoints.cpu.size(); p++) {
            cudaMallocManaged(&gpu_queries[q].gpu, dim * sizeof(float));
            for(int d = 0; d < dim; d++) {
                gpu_queries[q].gpu[d] = groupPoints[g].memberPoints.cpu.at(p).cpu.at(d);
            }
            queryIds[q] = groupPoints[g].queryIds.at(p);
            traverseRoot[q++] = groupPoints[g].traverseNode;
        }
    }

    // allocate storage of result
    cudaMallocManaged(&gpu_nearest, numQueries * sizeof(Point));
    for(int p = 0; p < numQueries; p++) {
        cudaMallocManaged(&gpu_nearest[p].gpu, dim * sizeof(float));
    }
    
    // for(int qsize = numQueries; qsize <= numQueries; qsize *= 2) {
    int qsize = numQueries;
    time_t start = clock();
    if(qsize > 512) runnable<<<qsize / 512, 512>>>(buffer.content, gpu_queries, queryIds, traverseRoot, gpu_nearest, qsize, dim);
    else runnable<<<1, qsize>>>(buffer.content, gpu_queries, queryIds, traverseRoot, gpu_nearest, qsize, dim);
    cudaDeviceSynchronize();
    time_t end = clock();
    cout << " Time taken for " << qsize << " using GPU kNN: " << (float)(end-start) / CLOCKS_PER_SEC << endl;
    // }

    // copy results to host memory
    for(int p = 0; p < numQueries; p++) {
        Point newPoint;
        for(int d = 0; d < dim; d++) newPoint.cpu.push_back(gpu_nearest[p].gpu[d]);
        cpu_nearest.push_back(newPoint);
        cudaFree(&gpu_nearest[p].gpu);
    }

    // deallocate memory
    for(int q = 0; q < numQueries; q++) {
        cudaFree(&gpu_queries[q].gpu);
    }
    cudaFree(&gpu_queries);

    for(int i = 0; i < buffer.pointer; i++) {
        // copy point info, starting and ending points into gpu
        if(buffer.get(i)->points.len > 0) cudaFree(&buffer.get(i)->points.gpu);
        cudaFree(&buffer.get(i)->startPoint.gpu);
        cudaFree(&buffer.get(i)->endPoint.gpu);
    }
}
