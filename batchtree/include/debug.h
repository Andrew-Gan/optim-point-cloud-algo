#include "structs.h"
#include <iostream>

using namespace std;

void debug_pretraversal(Node* root) {
    if(root == NULL) return;
    for(int i = 0; i < 2; i++) cout << root->startPoint.cpu.at(i) << " ";
    cout << endl;
    debug_pretraversal(root->left.cpu);
    debug_pretraversal(root->right.cpu);
}

void debug_gpu_pretraversal(Node root, Buffer* buff) {
    for(int i = 0; i < 2; i++) cout << root.startPoint.gpu[i] << " ";
    cout << endl;
    if(root.left.gpu == -1 && root.right.gpu == -1) return;
    debug_gpu_pretraversal(*(buff->get(root.left.gpu)), buff);
    debug_gpu_pretraversal(*(buff->get(root.right.gpu)), buff);
}