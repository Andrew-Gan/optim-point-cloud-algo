#include <fstream>
#include <iomanip>
#include "include/structs.h"
#include "include/hybridTree.h"

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
    ofs << "parent = " << right << setw(3) << currNode->points.cpu.size();
    ofs << ", left = " << right << setw(3) << currNode->left.cpu->points.cpu.size();
    ofs << ", right = " << right << setw(3) << currNode->right.cpu->points.cpu.size() << endl;
    // clear vector of points after usage and add interior nodes
    currNode->cleanNode();
}

/**************************************************
 * Recursive call for tree construction
**************************************************/
void _recurseTree(Node* currNode, size_t* nodeCount) {
    if(currNode->points.cpu.size() <= 1) return;
    (*nodeCount) += 2;
    currNode->partitionSubspace();
    _printNode(currNode);
    _recurseTree(currNode->left.cpu, nodeCount);
    _recurseTree(currNode->right.cpu, nodeCount);
}

/**************************************************
 * Taking the list of points and the maximum range
 * of space, construct and return a tree structure.
**************************************************/
Node* buildTree(vector<Point> inputData, int max, float enlargement_factor, float rou_factor, size_t* nodeCount) {
    Node* rootNode = new Node();
    for(int i = 0; i < inputData.at(0).cpu.size(); i++) {
        rootNode->startPoint.cpu.push_back(0);
        rootNode->endPoint.cpu.push_back(max);
    }
    (*nodeCount)++;
    Node::enlargementFactor = enlargement_factor;
    Node::rou_factor = rou_factor;
    rootNode->points.cpu = inputData;
    _recurseTree(rootNode, nodeCount);
    ofs.close();
    return rootNode;
}

/**************************************************
 * Deallocate memory for the entire tree.
**************************************************/
void freeTree(Node* root) {
    if(root == nullptr) {return;}
    freeTree(root->left.cpu);
    freeTree(root->right.cpu);
    delete(root);
}