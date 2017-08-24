GCC=gcc -w -O3 -fopenmp -std=gnu99
VERSION=0.04

all:
	$(GCC) backup1.c -lm -o backpropagation

clean:
	rm -f *~ *.o *.bin backpropagation

release:
	tar -cvf backpropagation-code-$(VERSION).tar *.c *.h *m *sh Makefile
