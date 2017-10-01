GCC=g++ -w -std=c++11
NVCC=nvcc -arch=sm_20
OBJPATH=bin/obj
ThreadPool_lib=threadpool
Build_dir=build
VERSION=0.04


all:
	$(NVCC) $(ThreadPool_lib)/CondVar.cpp -c -o $(Build_dir)/CondVar.o 
	$(NVCC) $(ThreadPool_lib)/Mutex.cpp -c -o $(Build_dir)/Mutex.o 
	$(NVCC) $(ThreadPool_lib)/Task.cpp -c -o $(Build_dir)/Task.o 
	$(NVCC) $(ThreadPool_lib)/ThreadPool.cpp -c -o $(Build_dir)/ThreadPool.o
	$(NVCC) bpnet.cpp -c -o $(Build_dir)/bpnet.o -x cu
	$(NVCC) example.cpp -c -o $(Build_dir)/example.o
	$(NVCC) $(Build_dir)/CondVar.o $(Build_dir)/Mutex.o $(Build_dir)/Task.o $(Build_dir)/ThreadPool.o $(Build_dir)/bpnet.o $(Build_dir)/example.o  -o backpropagation
	rm -f *.o

	#$(GCC) CondVar.o Mutex.o Task.o ThreadPool.o  $(ThreadPool_lib)/threadpool_test.cpp -lpthread -o threadpool_test

clean:
	rm -f *~ *.o *.bin backpropagation
