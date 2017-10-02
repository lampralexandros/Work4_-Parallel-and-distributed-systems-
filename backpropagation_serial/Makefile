GCC=g++ -w -std=gnu99
VERSION=0.04

all:
	$(GCC) bpnet.cpp bpnet.h example.cpp -lm -o backpropagation

clean:
	rm -f *~ *.o *.bin backpropagationcpp

release:
	tar -cvf backpropagationcpp-code-$(VERSION).tar *.c *.h *m *sh Makefile
