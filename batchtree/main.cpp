#include <fstream>
#include <iostream>
#include "include/structs.h"
#include "include/genInput.h"
#include "include/hybridTree.h"
#include "include/cluster.h"

void print_points_to_file(vector<Point> points, ofstream& ofs) {
    for(int p = 0; p < points.size(); p++) {
        for(int d = 0; d < points.at(p).cpu.size(); d++) {
            ofs << points.at(p).cpu.at(d) << " ";
        }
        ofs << endl;
    }
}

int mypow(int base, int exp) {
    int val = 1;
    for(int i = 0; i < exp; i++) {
        val *= base;
    }
    return val;
}

int main(int argc, char** argv) {
    if(argc != 5) {
        cout << "Correct usage: ./hybrid_sp_tree <num of points> <num of dimensions> <num of queries> <two digit truncate count>" << endl;
        return EXIT_FAILURE;
    }

    int max = 10000;
    int dataNum = atoi(argv[1]);
    int dim = atoi(argv[2]);
    int queryNum = atoi(argv[3]);
    int truncateNode = atoi(argv[4]);
    int clusterUnitLen = 500;
    int batchNum = mypow((max / clusterUnitLen), dim);
    const char* filetosave = "data/data_nearestPointList";
    char actualFileToSave[30];

    for(int i = 0; i < 26; i++) {
        actualFileToSave[i] = filetosave[i];
    }
    for(int i = 26; i < 28; i++) {
        actualFileToSave[i] = argv[4][i -26];
    }
    actualFileToSave[28] = '\0';

    cout << "saving to file " << actualFileToSave << endl;

    vector<Point> inputData = genInput(dataNum, dim, max, false);
    vector<Point> query = genInput(queryNum, dim, max, true);

    size_t nodeCount = 0;
    Node* root = buildTree(inputData, max, 0.2, 0.8, &nodeCount);
    
    Buffer buff(nodeCount);
    buff.insert(*root);
    vector<Point> nearestPoints;
    mapTreeToBuffer(buff);
    GroupPoint* groupPoints = new GroupPoint[batchNum];
    groupQueries(query, groupPoints, clusterUnitLen);
    ofstream ofs(actualFileToSave, ios::out);

    traverseBatches(buff.content, groupPoints, batchNum, dim, truncateNode);
    kNN_gpu(buff, groupPoints, nearestPoints, dim, queryNum, batchNum);
    delete[] groupPoints;

    print_points_to_file(nearestPoints, ofs);
    ofs.close();
    freeTree(root);
    return EXIT_SUCCESS;
}