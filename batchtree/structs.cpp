#include <algorithm>
#include "include/structs.h"

using namespace std;

// Initialization of static class variables
float Node::enlargementFactor = 0;
float Node::rou_factor = 0;

/**************************************************
 * Initialize left and right pointers to null.
**************************************************/
Node::Node() {
    left.cpu = nullptr;
    right.cpu = nullptr;
}

/**************************************************
 * Given the dimension along which to partition the 
 * subspace, return the value of the midpoint which
 * will be used for partitioning the subspace evenly.
 * The midpoint is added to the interior node.
**************************************************/
int Node::getPartitionPlane(int dim) {
    float* dimArr = new float[points.cpu.size()];
    for(int p = 0; p < points.cpu.size(); p++) {
        dimArr[p] = points.cpu.at(p).cpu.at(dim);
    }
    // sort points
    sort(dimArr, dimArr + points.cpu.size());
    int partition = dimArr[points.cpu.size() / 2];
    for(int p = 0; p < points.cpu.size(); p++) {
        if(points.cpu.at(p).cpu.at(dim) == partition) {
            tmp.push_back(points.cpu.at(p));
            points.cpu.erase(points.cpu.begin() + p);
        }
    }
    delete[] dimArr;
    return partition;
}

/**************************************************
 * This function is typically called by parent node
 * to assign points to child nodes by taking pointList
 * and checking if each point falls within range
**************************************************/
void Node::receivePointsFromParent(vector<Point> pointList) {
    for(int p = 0; p < pointList.size(); p++) {
        bool isInside = true;
        for(int i = 0; isInside && i < startPoint.cpu.size(); i++) {
            isInside = pointList.at(p).cpu.at(i) >= startPoint.cpu.at(i) && pointList.at(p).cpu.at(i) <= endPoint.cpu.at(i);
        }
        if(isInside) {points.cpu.push_back(pointList.at(p));}
    }
}

/**************************************************
 * This function is called after a spill-tree is built.
 * Computes fraction of number of points in either child
 * over number of points in parent and determines if
 * the fraction exceeds the rou_factor.
 * If true, a metric tree replaces the spill tree.
**************************************************/
void Node::checkIfMetricIsBetter(int dim) {
    if((float)left.cpu->points.cpu.size() / points.cpu.size() > rou_factor || (float)right.cpu->points.cpu.size() / points.cpu.size() > rou_factor) {
        // reset state of child subspaces
        left.cpu->points.cpu.clear();
        right.cpu->points.cpu.clear();
        nodeType = METRIC_NODE;
        left.cpu->endPoint.cpu.at(dim) /= (1 + enlargementFactor);
        right.cpu->startPoint.cpu.at(dim) /= (1 - enlargementFactor);
        left.cpu->receivePointsFromParent(points.cpu);
        right.cpu->receivePointsFromParent(points.cpu);
    }
}

/**************************************************
 * Determines the dimension of partition using rotation
 * method and partitions node into two child nodes.
 * The boundaries of the two child nodes are calculated
 * and points are assigned to the children.
 * Constructs metric tree based on rou factor.
**************************************************/
void Node::partitionSubspace() {
    // circulate partition dimensions
    left.cpu = new Node(), right.cpu = new Node();
    left.cpu->partitionDim = (partitionDim + 1) % startPoint.cpu.size();
    right.cpu->partitionDim = (partitionDim + 1) % startPoint.cpu.size();
    // calculate new boundaries
    left.cpu->startPoint.cpu = startPoint.cpu;
    left.cpu->endPoint.cpu = endPoint.cpu;
    int partitionPlane = getPartitionPlane(partitionDim);
    left.cpu->endPoint.cpu.at(partitionDim) = (float)partitionPlane * (1 + enlargementFactor);
    right.cpu->startPoint.cpu = startPoint.cpu;
    right.cpu->startPoint.cpu.at(partitionDim) = (float)partitionPlane * (1 - enlargementFactor);
    right.cpu->endPoint.cpu = endPoint.cpu;
    // determine points that are in left and right subspace
    left.cpu->receivePointsFromParent(points.cpu);
    right.cpu->receivePointsFromParent(points.cpu);
    checkIfMetricIsBetter(partitionDim);
}

/**************************************************
 * Clear all points stored in subspace except for
 * the median point.
**************************************************/
void Node::cleanNode() {
    points.cpu.clear();
    for(int i = 0; i < tmp.size(); i++) {
        points.cpu.push_back(tmp.at(i));
    }
    tmp.clear();
}