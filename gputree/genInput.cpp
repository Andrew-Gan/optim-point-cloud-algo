#include <iostream>
#include <fstream>
#include "include/structs.h"
#include "include/genInput.h"

using namespace std;

/**************************************************
 * Given the number of points, dimensions, and 
 * max range, generate and return a vector of points.
**************************************************/
vector<Point> genInput(int numPoints, int numDim, int max, bool isQuery) {
    // use previous seed to generate next seed
    static int seed = rand();
    srand(seed);
    seed ^= rand();
    ofstream ofs;
    if(isQuery) {ofs.open("data/data_pointList.txt", ios::app);}
    else {ofs.open("data/data_pointList.txt", ios::out);}
    vector<Point> inputData;
    cout << "Generating point info to pointList.txt ..." << endl;
    if(isQuery) {ofs << "Query data:" << endl;}
    else {ofs << "Input data: " << endl;}
    // for every point, generate integer for each dimension
    for(int n = 0; n < numPoints; n++) {
        Point newPoint;
        for(int i = 0; i < numDim; i++) {
            newPoint.cpu.push_back(rand() % max);
            ofs << newPoint.cpu.at(i);
            ofs << " ";
        }
        // add new point into input data vector
        inputData.push_back(newPoint);
        ofs << endl;
        newPoint.cpu.clear();
    }
    ofs.close();
    return inputData;
}