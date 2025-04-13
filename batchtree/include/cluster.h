#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include "structs.h"
void groupQueries(vector<Point> &queries, GroupPoint* batchQueries, int unitLen);
void traverseBatches(Node* tree, GroupPoint* batchQueries, int numBatches, int dim, int truncateRounds);

#endif