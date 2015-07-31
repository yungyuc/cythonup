TARGET = fibc fibcy cfib

all: $(TARGET)

fibcy: fibcy.pyx
	cython --embed -a $<
	g++ -O3 -march=native -o $@ $@.c `python2-config --cflags --ldflags`

cfib: cfib.pyx
	python setup.py build_ext -if

%: %.c
	gcc -std=c11 -O3 -march=native -o $@ $@.c

.PHONY: clean
clean:
	@rm -f $(TARGET) *.o *.html *.so fibcy.c cfib.c
	@rm -fr build
