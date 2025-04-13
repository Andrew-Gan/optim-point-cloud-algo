#include <fstream>
#include <iomanip>
#include "structs.h"

using namespace std;

// recommended values
// enlargement_factor = 0.2;  // factor to enlarge boundaries of all subspaces
// rou_factor = 0.75;         // threshold to determine SP or Metric (higher value -> more likely to use SP)
ofstream ofs("data/data_treeNodes.txt");  // file that contains node info

/**************************************************
 * This function should never receive a leaf node.
 * Prints out node type, number of points in self, 
 * left, right and clear points in received node.
**************************************************/
void _printNode(Node* currNode) {
    switch(currNode->nodeType) {
        case 0 : ofs << "metric ";
            break;
        case 1 : ofs << "spill  ";
    }
    ofs << "parent = " << right << setw(3) << currNode->points.size();
    ofs << ", left = " << right << setw(3) << currNode->left->points.size();
    ofs << ", right = " << right << setw(3) << currNode->right->points.size() << endl;
    // clear vector of points after usage and add interior nodes
    currNode->cleanNode();
}

/**************************************************
 * Recursive call for tree construction
**************************************************/
void _recurseTree(Node* currNode) {
    if(currNode->points.size() <= 1) {return;}
    currNode->partitionSubspace();
    _printNode(currNode);
    _recurseTree(currNode->left);
    _recurseTree(currNode->right);
}

/**************************************************
 * Taking the list of points and the maximum range
 * of space, construct and return a tree structure.
**************************************************/
Node* buildTree(vector<Point> inputData, int max, float enlargement_factor, float rou_factor) {
    Node* rootNode = new Node();
    for(int i = 0; i < inputData.at(0).size(); i++) {
        rootNode->startPoint.push_back(0);
        rootNode->endPoint.push_back(max);
    }
    Node::enlargementFactor = enlargement_factor;
    Node::rou_factor = rou_factor;
    rootNode->points = inputData;
    _recurseTree(rootNode);
    ofs.close();
    return rootNode;
}

/**************************************************
 * Deallocate memory for the entire tree.
**************************************************/
void freeTree(Node* root) {
    if(root == nullptr) return;
    freeTree(root->left);
    freeTree(root->right);
    delete(root);
}