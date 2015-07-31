TARGET = helloworld

all: $(TARGET)

%: %.pyx
	@echo "\nConverting to c code....."
	cython --embed -a $<
	@echo "\nBuilding binary.........."
	g++ -O3 -march=native -o $@ $@.c `python2-config --cflags --ldflags`
	@echo "\n"

lib:
	@echo "\nBuilding as a shared library"
	python setup.py build_ext --inplace -f  # -if
	@echo "\n"

.PHONY: clean
clean:
	@rm -f *.c $(TARGET) *.o *.html *.so
