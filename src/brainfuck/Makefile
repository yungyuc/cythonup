TARGET = bf0 bf1 bf2 bf3 bf4 bf5

all: $(TARGET)

%.cpp: %.pyx
	cython --embed --cplus -a $<

%.cpp: %.py
	cython --embed --cplus -a $<

bf%:	
	g++ -O2 -o $@ $@*.cpp `python2-config --cflags --ldflags`

bf0: bf0_original.cpp
bf1: bf1_map.cpp
bf2: bf2_vector.cpp
bf3: bf3_vector_all.cpp
bf4: bf4_merge.cpp
bf5: bf5_improved.cpp

.PHONY: clean
clean:
	rm -f *.cpp $(TARGET) *.html