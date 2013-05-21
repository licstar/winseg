CC = g++
GCC = gcc
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: sennaseg

sennaseg : sennaseg.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) sennaseg.cpp -fopenmp -DLINUX -o sennaseg

clean:
	rm -rf *.o sennaseg
