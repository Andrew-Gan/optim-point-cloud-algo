#ifndef __KNN_GPU__H__
#define __KNN_GPU__H__

__global__
void runnable(Node* buffer, Point* queries, Point* nearest, int qsize, int dim);

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

/**************************************************
 * This function should be called after root node
 * is inserted at pos 0 of the buffer.
 * Given a buffer with the root node loaded to 
 * offset 0, map the tree nodes into an array
 * structure in breadth-first order and replace
 * pointers to children with offset in array.
**************************************************/
__host__
void mapTreeToBuffer(Buffer& buffer) {
    if(buffer.pointer != 1) return;

    for(int i = 0; i < buffer.capacity; i++) {
        // store child nodes in available space
        if(buffer.get(i)->left.cpu != nullptr && buffer.get(i)->right.cpu != nullptr) {
            buffer.get(i)->left.gpu = buffer.insert(*(buffer.get(i)->left.cpu));
            buffer.get(i)->right.gpu = buffer.insert(*(buffer.get(i)->right.cpu));
        }
        else {
            buffer.get(i)->left.gpu = -1;
            buffer.get(i)->right.gpu = -1;
        }
    }
}

#endif