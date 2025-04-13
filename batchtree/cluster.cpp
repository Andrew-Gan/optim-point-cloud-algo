#include "include/structs.h"
#include <iostream>

// buffer passed it is before gpu mapping
void groupQueries(vector<Point> &queries, GroupPoint* batchQueries, int unitLen) {
    int batchIndex;
    int dim = queries.at(0).cpu.size();

    for(int i = 0; i < queries.size(); i++) {
        batchIndex = 0;
        for(int d = 0; d < dim; d++) {
            batchIndex *= 10000 / unitLen;
            batchIndex += queries.at(i).cpu.at(d) / unitLen;
        }
        // std::cout << "batchIndex = " << batchIndex << " " << "query = " << i << std::endl;
        // register point into batch
        batchQueries[batchIndex].queryIds.push_back(i);
        batchQueries[batchIndex].memberPoints.cpu.push_back(queries.at(i));

        // init starting and ending point if not set
        if(batchQueries[batchIndex].memberPoints.cpu.size() == 1) {
            batchQueries[batchIndex].startPoint = new int[dim];
            batchQueries[batchIndex].endPoint = new int[dim];
            for(int d = 0; d < dim; d++) {
                batchQueries[batchIndex].startPoint[d] = queries.at(i).cpu.at(d);
                batchQueries[batchIndex].endPoint[d] = queries.at(i).cpu.at(d);
            }
        }
        // update starting and ending point if set previously
        else {
            for(int d = 0; d < dim; d++) {
                if (batchQueries[batchIndex].startPoint[d] > queries.at(i).cpu.at(d)) {
                    batchQueries[batchIndex].startPoint[d] = queries.at(i).cpu.at(d);
                }
                if (batchQueries[batchIndex].endPoint[d] < queries.at(i).cpu.at(d)) {
                    batchQueries[batchIndex].endPoint[d] = queries.at(i).cpu.at(d);
                }
            }
        }
    }
}

void traverseBatches(Node* tree, GroupPoint* batchQueries, int numBatches, int dim, int truncateRounds) {
    Node currNode;
    int currNodeHandle;
    bool isIn = true;

    for(int i = 0; i < numBatches; i++) {
        if(batchQueries[i].memberPoints.cpu.size() > 0) {
            currNode = tree[0];
            currNodeHandle = 0;
            isIn = true;
            for(int r = 0; r < truncateRounds && isIn; r++) {
                // check if left child completely contains batch
                for(int d = 0; d < dim && isIn; d++) {
                    bool s = batchQueries[i].startPoint[d] > tree[currNode.left.gpu].startPoint.cpu.at(d);
                    bool e = batchQueries[i].endPoint[d] < tree[currNode.left.gpu].endPoint.cpu.at(d);
                    isIn = s && e;
                }
                if(isIn) {
                    currNodeHandle = currNode.left.gpu;
                    currNode = tree[currNode.left.gpu];
                }
                if(!isIn) {
                    isIn = true;
                    // check if right child completely contains batch
                    for(int d = 0; d < dim && isIn; d++) {
                        bool s = batchQueries[i].startPoint[d] > tree[currNode.right.gpu].startPoint.cpu.at(d);
                        bool e = batchQueries[i].endPoint[d] < tree[currNode.right.gpu].endPoint.cpu.at(d);
                        isIn = s && e;
                    }
                    if(isIn) {
                        currNodeHandle = currNode.right.gpu;
                        currNode = tree[currNode.right.gpu];
                    }
                }
            }
            batchQueries[i].traverseNode = currNodeHandle;
            cout << "Batch " << i << " with numPoints " << batchQueries[i].memberPoints.cpu.size() << " has chosen node " << currNodeHandle << endl;
        }
    }
}