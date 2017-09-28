GCC=g++ -w -std=gnu99 
path_to_boost=/usr/local/boost_1_65_1
VERSION=0.04

all:
	$(GCC) bpnet.cpp bpnet.h example.cpp -lm -I $(path_to_boost) -o backpropagation 

clean:
	rm -f *~ *.o *.bin backpropagationcpp

release:
	tar -cvf backpropagationcpp-code-$(VERSION).tar *.c *.h *m *sh Makefile
