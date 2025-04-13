#include <algorithm>
#include "structs.h"
#include <iostream>

using namespace std;

// Initialization of static class variables
float Node::enlargementFactor = 0;
float Node::rou_factor = 0;

/**************************************************
 * Given the dimension along which to partition the 
 * subspace, return the value of the midpoint which
 * will be used for partitioning the subspace evenly.
 * The midpoint is added to the interior node.
**************************************************/
int Node::getPartitionPlane(int dim) {
    float* dimArr = new float[points.size()];
    for(int p = 0; p < points.size(); p++) {
        dimArr[p] = points.at(p).at(dim);
    }
    // sort points
    sort(dimArr, dimArr + points.size());
    int partition = dimArr[points.size() / 2];
    for(int p = 0; p < points.size(); p++) {
        if(points.at(p).at(dim) == partition) {
            tmp.push_back(points.at(p));
            points.erase(points.begin() + p);
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
        for(int i = 0; isInside && i < startPoint.size(); i++) {
            isInside = pointList.at(p).at(i) >= startPoint.at(i) && pointList.at(p).at(i) <= endPoint.at(i);
        }
        if(isInside) {points.push_back(pointList.at(p));}
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
    if((float)left->points.size() / points.size() > rou_factor || (float)right->points.size() / points.size() > rou_factor) {
        // reset state of child subspaces
        left->points.clear();
        right->points.clear();
        nodeType = METRIC_NODE;
        left->endPoint.at(dim) /= (1 + enlargementFactor);
        right->startPoint.at(dim) /= (1 - enlargementFactor);
        left->receivePointsFromParent(points);
        right->receivePointsFromParent(points);
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
    left = new Node(), right = new Node();
    left->dim = (dim + 1) % startPoint.size();
    right->dim = (dim + 1) % startPoint.size();
    // calculate new boundaries
    left->startPoint = startPoint;
    left->endPoint = endPoint;
    int partitionPlane = getPartitionPlane(dim);
    left->endPoint.at(dim) = (float)partitionPlane * (1 + enlargementFactor);
    right->startPoint = startPoint;
    right->startPoint.at(dim) = (float)partitionPlane * (1 - enlargementFactor);
    right->endPoint = endPoint;
    // determine points that are in left and right subspace
    left->receivePointsFromParent(points);
    right->receivePointsFromParent(points);
    checkIfMetricIsBetter(dim);
}

/**************************************************
 * Given a Point type, determine whether the point
 * is inside subspace and returns a boolean value.
**************************************************/
bool Node::checkIfInsideSubspace(Point query) {
    for(int dim = 0; dim < query.size(); dim++) {
        if(query.at(dim) < startPoint.at(dim) || query.at(dim) > endPoint.at(dim)) {
            return false;
        }
    }
    return true;
}
/**************************************************
 * Clear all points stored in subspace except for
 * the median point.
**************************************************/
void Node::cleanNode() {
    points.clear();
    for(int i = 0; i < tmp.size(); i++) {
        points.push_back(tmp.at(i));
    }
    tmp.clear();
}