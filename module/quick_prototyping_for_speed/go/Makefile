TARGET = go

all: $(TARGET)

%: %.py
	cython --embed -a $<
	g++ -O2 -o $@ $@.c `python2-config --cflags --ldflags`


.PHONY: clean
clean:
	rm -f *.c $(TARGET) *.html