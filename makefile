libbm25.so: bm25.o
	g++ -shared -pthread -Wl,-soname,libbm25.so -o libbm25.so bm25.o
	rm -f *.o

bm25.o:
	g++ -c -fPIC bm25.cpp --std=c++11 -o bm25.o

clean:
	rm -f *.o *.so


.PHONY: clean

