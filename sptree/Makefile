SOURCE_CPP = generator/genInput.cpp hybridTree.cpp structs.cpp main.cpp
KNN_CPP = kNN_autorope.cpp kNN_traversal.cpp kNN_parallel.cpp kNN_guaranteed.cpp
PLOT_PY = visualization/PlotPoints.py
SOURCE_H = structs.h
COMPILER = g++
OUTPUT = main
FLAGS = -ltbb

build: $(SOURCE_H)
	$(COMPILER) -g -o $(OUTPUT) $(SOURCE_CPP) $(KNN_CPP) $(FLAGS)

run: $(OUTPUT)
	./$(OUTPUT) 10000 2 32768

plot: 
	cd visualization; make

verify:
	cd verification; make
	
clean:
	rm -f $(OUTPUT) data/*
