#ifndef __HYBRID_TREE_H__
#define __HYBRID_TREE_H__

Node* buildTree(vector<Point>, int, float, float, size_t*);
void freeTree(Node*);
void mapTreeToBuffer(Buffer&);

#endif