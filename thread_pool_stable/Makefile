GCC=g++ -w -std=c++11
OBJPATH=bin/obj
ThreadPool_lib=threadpool
VERSION=0.04

all:
	$(GCC) $(ThreadPool_lib)/CondVar.cpp -lpthread -c -o CondVar.o
	$(GCC) $(ThreadPool_lib)/Mutex.cpp -lpthread -c -o Mutex.o
	$(GCC) $(ThreadPool_lib)/Task.cpp -lpthread -c -o Task.o
	$(GCC) $(ThreadPool_lib)/ThreadPool.cpp -lpthread -c -o ThreadPool.o
	$(GCC) CondVar.o Mutex.o Task.o ThreadPool.o bpnet.cpp bpnet.h example.cpp -o backpropagation -lm  -pthread

	#$(GCC) CondVar.o Mutex.o Task.o ThreadPool.o  $(ThreadPool_lib)/threadpool_test.cpp -lpthread -o threadpool_test

clean:
	rm -f *~ *.o *.bin backpropagationcpp
