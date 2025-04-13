#ifndef __KNN_GPU__H__
#define __KNN_GPU__H__

__global__
void runnable(Node* buffer, Point* queries, int* queryIds, int* traverseRoot, Point* nearest, int qsize, int dim);

__device__
bool checkIfInsideSubspace(float* startPosArr, float* endPosArr, float* query, unsigned int dim);

struct Stack {
    int content[20];
    int offset = 0;
};

__device__
void push(Stack& s, int i) {
    s.content[s.offset++] = i;
}

__device__
int pop(Stack& s) {
    return s.content[--s.offset];
}

__device__
bool isEmpty(Stack& s) {
    return s.offset == 0;
}

/**************************************************
 * Initialize a unique pointer to allocated memory
 * and initialize fields.
**************************************************/
Buffer::Buffer(unsigned int i) {
    capacity = i;
    pointer = 0;
    cudaMallocManaged(&content, i * sizeof(Node));
}

/**************************************************
 * Deallocate memory from buffer.
**************************************************/
Buffer::~Buffer() {
    cudaFree(content);
}

/**************************************************
 * Insert a new element to the end of the buffer.
**************************************************/
int Buffer::insert(Node newNode) {
    if(pointer >= capacity) return -1;
    content[pointer++] = newNode;
    return pointer - 1;
}

/**************************************************
 * Receive offset and return pointer of stored node.
 * Returns a null pointer if offset is invalid.
**************************************************/
Node* Buffer::get(unsigned int offs) {
    if(offs >= pointer) return nullptr;
    return &(content[offs]);
}

#endif