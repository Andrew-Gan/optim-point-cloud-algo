#ifndef __STRUCTS_H__
#define __STRUCTS_H__

#include <vector>

using namespace std;

enum NodeType {METRIC_NODE, SPILL_NODE};

class Node;

struct Point {
    vector<float> cpu;
    float* gpu;
};

struct Points {
    vector<Point> cpu;
    unsigned int len;
    float* gpu;
};

union Child {
    Node* cpu;
    int gpu;
};

class Node {
public:
    // class fields
    int partitionDim = 0;               // dimension of partition (circulate)
    static float enlargementFactor;     // range of enlargement
    static float rou_factor;            // threshold to determine if SP or Metric should be used
    Points points;                      // points belonging to node
    NodeType nodeType = SPILL_NODE;     // type of node, metric or spill
    Point startPoint, endPoint;         // two points that determine boundary
    Child left;                         // pointer to left child
    Child right;                        // pointer to right child
    // class functions
    Node();
    virtual ~Node() {}
    void partitionSubspace();
    void cleanNode();
private:
    vector<Point> tmp;
    int getPartitionPlane(int);
    void receivePointsFromParent(vector<Point>);
    void checkIfMetricIsBetter(int);
};

class Buffer {
public:
    Node* content;
    unsigned int capacity;
    unsigned int pointer;
    Buffer(unsigned int);
    // Buffer(const Buffer&);
    virtual ~Buffer();
    int insert(Node newNode);
    Node* get(unsigned int);
};

struct GroupPoint {
    int traverseNode;
    int* startPoint;
    int* endPoint;
    vector<int> queryIds;
    Points memberPoints;
};

void kNN_gpu(Buffer buffer, GroupPoint* groupPoints, vector<Point>& cpu_nearest, int dim, int numQueries, int numGroups);

#endif