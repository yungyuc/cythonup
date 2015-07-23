TARGET = go

all: $(TARGET)

%: %.py
	cython --embed -a $<
	g++ -Ofast -march=native -o $@ $@.c `python2-config --cflags --ldflags`


.PHONY: clean
clean:
	rm -f *.c $(TARGET) *.html