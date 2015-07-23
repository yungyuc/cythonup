TARGET = bf0 bf1 bf2 bf3 bf4 bfx4 bf5 bfx5 bff4

all: $(TARGET)

%.cpp: %.pyx
	cython --embed --cplus -a $<

%.cpp: %.py
	cython --embed --cplus -a $<

bf%:	
	g++ -O3 -march=native -o $@ $@*.cpp `python2-config --cflags --ldflags`
bff4: bff4.c
	gcc -O3 -march=native -o $@ bff4.c

bf0: bf0_original.cpp bf0_original.py
bf1: bf1_map.cpp bf1_map.pyx
bf2: bf2_vector.cpp bf2_vector.pyx
bf3: bf3_vector_all.cpp bf3_vector_all.pyx
bf4: bf4_merge.cpp bf4_merge.py bf4_merge.pxd
bf5: bf5_improved.cpp bf5_improved.py bf5_improved.pxd
bfx5: bfx5.cpp bfx5.pyx
bfx5.pyx: bf5_improved.py bf5_improved.pxd
	cp bf5_improved.pxd bfx5.pyx
	tail -n +4 bf5_improved.py >> bfx5.pyx

bfx4: bfx4.cpp bfx4.pyx
bfx4.pyx: bf4_merge.py bf4_merge.pxd
	head -1 bf4_merge.py > bfx4.pyx
	cat bf4_merge.pxd >> bfx4.pyx
	tail -n +2 bf4_merge.py >> bfx4.pyx	

.PHONY: clean
clean:
	rm -f *.cpp $(TARGET) *.html bfx5.pyx