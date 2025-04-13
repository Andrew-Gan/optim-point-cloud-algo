#include "structs.h"

void _recurseDFS(Node* parent, Point query, Point* nearestPoint, long* min_dist) {
    for(int p = 0; p < parent->points.size(); p++) {
        long curr_dist = 0;
        for(int d = 0; d < query.size(); d++) {
            curr_dist += (query.at(d) - parent->points.at(0).at(d)) * (query.at(d) - parent->points.at(0).at(d));
        }
        if(curr_dist < *min_dist) {
            *min_dist = curr_dist;
            *nearestPoint = parent->points.at(p);
        }
    }

    // if non-leaf node, keep traversing
    if(parent->left != nullptr && parent->right != nullptr) {
        if(parent->nodeType == METRIC_NODE) {
            _recurseDFS(parent->left, query, nearestPoint, min_dist);
            _recurseDFS(parent->right, query, nearestPoint, min_dist);
        }
        else if (parent->nodeType == SPILL_NODE) {
            if(parent->left->checkIfInsideSubspace(query)) {_recurseDFS(parent->left, query, nearestPoint, min_dist);}
            if(parent->right->checkIfInsideSubspace(query)) {_recurseDFS(parent->right, query, nearestPoint, min_dist);}
        }
    }
}

Point kNN_traversal(Node* root, Point query) {
    long min_dist = __LONG_MAX__;
    Point point;
    _recurseDFS(root, query, &point, &(min_dist));
    return point;
}