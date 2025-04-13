#ifndef __SP_TREE_H__
#define __SP_TREE_H__

#include <vector>
using namespace std;

enum NodeType {METRIC_NODE, SPILL_NODE};
typedef vector<float> Point;

class Node {
public:
    // class fields
    int dim = 0;                        // dimension of partition (circulate)
    static float enlargementFactor;     // range of enlargement
    static float rou_factor;            // threshold to determine if SP or Metric should be used
    vector<Point> points;               // points belonging to node
    NodeType nodeType = SPILL_NODE;     // type of node, metric or spill
    Point startPoint, endPoint;         // two points that determine boundary
    Node* left = nullptr;               // pointer to left child
    Node* right = nullptr;              // pointer to right child
    // class functions
    Node() {}
    virtual ~Node() {}
    void partitionSubspace();
    bool checkIfInsideSubspace(Point);
    void cleanNode();
private:
    vector<Point> tmp;
    int getPartitionPlane(int);
    void receivePointsFromParent(vector<Point>);
    void checkIfMetricIsBetter(int);
};

class Stack {
private:
    Node* content[50];
    int offset = 0;
public:
    void push(Node* newNode) {content[offset++] = newNode;}
    Node* pop() {return content[--offset];}
    bool isEmpty() {return offset == 0;}
};

// genInput.cpp functions
vector<Point> genInput(int, int, int, bool);
// hybridTree.cpp functions
Node* buildTree(vector<Point>, int, float, float);
void freeTree(Node*);
// kNN functions
Point kNN_traversal(Node*, Point);
Point kNN_autorope(Node*, Point);
Point* kNN_parallel(Node*, vector<Point>, Point*, int);
void kNN_guaranteed(vector<Point> &queries, vector<Point> &data, short* nearest);

#endif