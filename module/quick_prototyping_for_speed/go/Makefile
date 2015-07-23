TARGET = go

all: $(TARGET)

%: %.py
	cython --embed -a $<
	g++ -O3 -march=native -o $@ $@.c `python2-config --cflags --ldflags`


.PHONY: clean
clean:
	rm -f *.c $(TARGET) *.html