#include "structs.h"

float _get_sqr_dist(Point a, Point b) {
    float dist = 0;
    for(int d = 0; d < a.size(); d++) {
        dist += (a.at(d) - b.at(d)) * (a.at(d) - b.at(d));
    }
    return dist;
}

void kNN_guaranteed(vector<Point> &queries, vector<Point> &data, short* nearest) {
    float min_dist = -1;
    float curr_dist = 0;
    int currMinHandle = 0;
    for(int q = 0; q < queries.size(); q++) {
        min_dist = -1;
        for(int d = 0; d < data.size(); d++) {
            curr_dist = _get_sqr_dist(queries.at(q), data.at(d));
            if(min_dist == -1) {
                min_dist = curr_dist;
                currMinHandle = d;
            }
            else if(min_dist > curr_dist) {
                min_dist = curr_dist;
                currMinHandle = d;
            }
        }
        nearest[q] = currMinHandle;
    }
}