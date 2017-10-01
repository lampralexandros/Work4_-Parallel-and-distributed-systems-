GCC=g++ -w -std=c++11
NVCC=nvcc -arch=sm_20
OBJPATH=bin/obj
ThreadPool_lib=threadpool
VERSION=0.04


all:
	$(NVCC) $(ThreadPool_lib)/CondVar.cpp -c -o CondVar.o 
	$(NVCC) $(ThreadPool_lib)/Mutex.cpp -c -o Mutex.o 
	$(NVCC) $(ThreadPool_lib)/Task.cpp -c -o Task.o 
	$(NVCC) $(ThreadPool_lib)/ThreadPool.cpp -c -o ThreadPool.o
	$(NVCC) bpnet.cpp -c -o bpnet.o -x cu
	$(NVCC) example.cpp -c -o example.o
	$(NVCC) CondVar.o Mutex.o Task.o ThreadPool.o bpnet.o example.o  -o backpropagation

	#$(GCC) CondVar.o Mutex.o Task.o ThreadPool.o  $(ThreadPool_lib)/threadpool_test.cpp -lpthread -o threadpool_test

clean:
	rm -f *~ *.o *.bin backpropagationcpp
