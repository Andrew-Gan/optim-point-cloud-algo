#include <fstream>
#include <iostream>
#include <time.h>
#include "structs.h"

void print_point_to_file(Point p, ofstream* ofs) {
    for(int dim = 0; dim < p.size(); dim++) {
        *ofs << p.at(dim) << " ";
    }
    *ofs << endl;
}

int main(int argc, char** argv) {
    if(argc != 4) {
        cout << "Correct usage: ./hybrid_sp_tree <num of points> <num of dimensions> <num of queries>" << endl;
        return EXIT_FAILURE;
    }
    int max = 10000;
    int qsize = atoi(argv[3]);
    vector<Point> inputData = genInput(atoi(argv[1]), atoi(argv[2]), max, false);
    vector<Point> query = genInput(qsize, atoi(argv[2]), max, true);
    short* nearestPoints = new short[qsize];
    const char* fileToSave = "data/data_nearestPoint";
    char actFileToSave[26];

    for(int i = 0; i < 22; i++) {
        actFileToSave[i] = fileToSave[i];
    }
    actFileToSave[25] = '\0';

    // kNN_guaranteed(query, inputData, nearestPoints);
    // for(int i = 0; i < qsize; i++) {
    //     print_point_to_file(inputData.at(nearestPoints[i]), &ofs);
    // }
    // ofstream timefs("data/data_timeTaken.txt", ios::out);

    // // traversal kNN
    // time_t start = clock();
    // for(int i = 0; i < qsize; i++) Point nearestPoint = kNN_traversal(root, query[i]);
    // time_t end = clock();
    // cout << "Time taken for traversal kNN: " << (float)(end-start) / CLOCKS_PER_SEC << endl;

    for(float rho = 0; rho < 1; rho += 0.01) {
        actFileToSave[22] = (((int)(rho * 100) / 100) % 10) + '0';
        actFileToSave[23] = (((int)(rho * 100) / 10) % 10) + '0';
        actFileToSave[24] = (((int)(rho * 100) / 1) % 10) + '0';
        ofstream ofs(actFileToSave, ios::out);
        Node* root = buildTree(inputData, max, 0.2, rho);

        cout << "Performing rho = " << rho << endl;

        // autoroping kNN
        // time_t start = clock();
        for(int i = 0; i < qsize; i++) {
            Point nearestPoint = kNN_autorope(root, query[i]);
            print_point_to_file(nearestPoint, &ofs);
        }
        ofs.close();
        freeTree(root);
        // time_t end = clock();
        // cout << "Time taken for " << q << " autorope kNN: " << (float)(end-start) / CLOCKS_PER_SEC << endl;
        // timefs << (float)(end-start) / CLOCKS_PER_SEC << endl;
    }

    // // parallel kNN
    // Point* nearestPoints = new Point[qsize];
    // for(int i = 8; i <= 16; i+=4) {
    //     start = clock();
    //     kNN_parallel(root, query, nearestPoints, i);
    //     end = clock();
    //     cout << "Time taken for parallel kNN (" << i << " threads) : " << (float)(end-start) / CLOCKS_PER_SEC << endl;
    // }
    // delete[] nearestPoints;

    // ofs.close();
    // timefs.close();
    // freeTree(root);
    return EXIT_SUCCESS;
}