#include <fstream>
#include <iostream>
#include "include/structs.h"
#include "include/genInput.h"
#include "include/hybridTree.h"

void print_points_to_file(vector<Point> points, ofstream& ofs) {
    for(int p = 0; p < points.size(); p++) {
        for(int d = 0; d < points.at(p).cpu.size(); d++) {
            ofs << points.at(p).cpu.at(d) << " ";
        }
        ofs << endl;
    }
}

int main(int argc, char** argv) {
    if(argc != 4) {
        cout << "Correct usage: ./hybrid_sp_tree <num of points> <num of dimensions> <num of queries>" << endl;
        return EXIT_FAILURE;
    }
    int max = 10000;
    vector<Point> inputData = genInput(atoi(argv[1]), atoi(argv[2]), max, false);
    vector<Point> query = genInput(atoi(argv[3]), atoi(argv[2]), max, true);
    size_t nodeCount = 0;
    Node* root = buildTree(inputData, max, 0.2, 0.8, &nodeCount);
    ofstream ofs("data/data_nearestPointList.txt", ios::out);
    Buffer buff(nodeCount);
    buff.insert(*root);
    vector<Point> nearestPoints;
    kNN_gpu(buff, query, nearestPoints, 16, 256);
    print_points_to_file(nearestPoints, ofs);
    ofs.close();
    freeTree(root);
    return EXIT_SUCCESS;
}