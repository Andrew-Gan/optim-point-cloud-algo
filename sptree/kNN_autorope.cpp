#include "structs.h"
#include <fstream>

// ofstream ofs2("data/data_subspace.txt");

Point kNN_autorope(Node* root, Point query) {
    long min_dist = __LONG_MAX__;
    long curr_dist;
    Point nearestPoint;
    Stack nodeStack;
    nodeStack.push(root);
    // keep examining tree until stack is empty
    while(!nodeStack.isEmpty()) {
        // pop root
        Node* root = nodeStack.pop();

        // // print visited node data to text file
        // for(int i = 0; i < root->startPoint.size(); i++) ofs2 << root->startPoint[i] << " ";
        // ofs2 << endl;
        // for(int i = 0; i < root->endPoint.size(); i++) ofs2 << root->endPoint[i] << " ";
        // ofs2 << endl;

        for(int p = 0; p < root->points.size(); p++) {
            curr_dist = 0;
            for(int d = 0; d < query.size(); d++) {
                curr_dist += (query.at(d) - root->points.at(p).at(d)) * (query.at(d) - root->points.at(p).at(d));
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
                if(root->right->checkIfInsideSubspace(query)) {nodeStack.push(root->right);}
                if(root->left->checkIfInsideSubspace(query)) {nodeStack.push(root->left);}
            }
        }
    }
    return nearestPoint;
}