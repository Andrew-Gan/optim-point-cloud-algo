#include <tbb/tbb.h>
#include "structs.h"

using namespace std;
using namespace tbb;

class Runnable {
public:
    static Node* root;
    static vector<Point> queryPoints;
    static Point* nearestPoints;

    void operator() (const blocked_range<size_t>&r) const {
        // time_t start = clock();
        for(int q = r.begin(); q != r.end(); q++) {
            long min_dist = __LONG_MAX__;
            long curr_dist;
            Point nearestPoint;
            Stack nodeStack;
            nodeStack.push(root);
            // keep examining tree until stack is empty
            while(!nodeStack.isEmpty()) {
                // pop root
                Node* root = nodeStack.pop();
                for(int p = 0; p < root->points.size(); p++) {
                    curr_dist = 0;
                    for(int d = 0; d < queryPoints[q].size(); d++) {
                        curr_dist += (queryPoints[q].at(d) - root->points.at(p).at(d)) * (queryPoints[q].at(d) - root->points.at(p).at(d));
                    }
                    if(curr_dist < min_dist) {
                        min_dist = curr_dist;
                        nearestPoint = root->points.at(p);
                    }
                }
                // if non-leaf node, keep traversing
                if(root->left != nullptr && root->right != nullptr) {
                    if(root->nodeType == METRIC_NODE) {
                        nodeStack.push(root->right);
                        nodeStack.push(root->left);
                    }
                    else if (root->nodeType == SPILL_NODE) {
                        if(root->right->checkIfInsideSubspace(queryPoints[q])) {nodeStack.push(root->right);}
                        if(root->left->checkIfInsideSubspace(queryPoints[q])) {nodeStack.push(root->left);}
                    }
                }
            }
            nearestPoints[q] = nearestPoint;
        }
        // time_t end = clock();
        // cout << "start: " << (float)start / CLOCKS_PER_SEC << " end: " << (float)end / CLOCKS_PER_SEC << " duration: " << (float)(end-start)/CLOCKS_PER_SEC << endl;
    }

    Runnable() {}
    Runnable(const Runnable& arg) {}
    virtual ~Runnable() {}
};

Node* Runnable::root = nullptr;
vector<Point> Runnable::queryPoints;
Point* Runnable::nearestPoints = nullptr;

Point* kNN_parallel(Node* root, vector<Point> queries, Point* nearestPoints, int nthread) {
    Runnable myRunnable;
    Runnable::root = root;
    Runnable::queryPoints = queries;
    Runnable::nearestPoints = nearestPoints;
    tbb::task_scheduler_init anonymous;
    parallel_for(blocked_range<size_t>(0, queries.size(), queries.size() / nthread), myRunnable);
    return nearestPoints;
}