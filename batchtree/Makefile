CPU_SOURCE = genInput.cpp hybridTree.cpp structs.cpp main.cpp cluster.cpp
GPU_SOURCE = kNN_gpu.cpp 
CPU_OBJECT = genInput.o hybridTree.o structs.o main.o cluster.o
GPU_OBJECT = kNN_gpu.o
PLOT_PY = visualization/PlotPoints.py
CPU_COMPILER = g++
GPU_COMPILER = nvcc
OUTPUT = main

# compiles CPU and GPU code separately, and links them together
# use g++ 	to compile 	host source 									-> host object
# use nvcc 	to compile 	device source + mixed source 					-> device object + mixed object
# use nvcc 	to link 	device source + device portion of mixed source 	-> arch object
# use g++ 	to link 	arch object + host object + host portion of mixed source -> executable
compile: $(CPU_SOURCE) $(GPU_SOURCE)
	$(CPU_COMPILER) -c -std=c++11 $(CPU_SOURCE)
	$(GPU_COMPILER) -dc -x cu $(GPU_SOURCE)
	$(GPU_COMPILER) -arch=sm_30 -dlink $(GPU_OBJECT) -o  gpuCode.o
	$(GPU_COMPILER) -I/usr/local/cuda/lib64/libcudart gpuCode.o $(CPU_OBJECT) $(GPU_OBJECT) -o $(OUTPUT)

# unload intel compiler and load gcc compiler
# load cuda libraries
load:
	module unload intel; \
	module load gcc; \
	module load cuda; \

run: $(OUTPUT)
	./$(OUTPUT) 10000 2 32768

verify:
	cd verification; make; cd ..

plot: 
	cd visualization; make; cd ..

git:
	git add .
	git commit -m "~"
	git push

send:
	scp Makefile *cpp gan35@scholar.rcac.purdue.edu:~/cuda
	scp -r include gan35@scholar.rcac.purdue.edu:~/cuda

receive:
	scp -r gan35@scholar.rcac.purdue.edu:~/cuda/data .

clean:
	rm -f $(OUTPUT) *.o data/*