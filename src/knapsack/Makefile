TARGET = knapsack_0_original knapsack_1_improved knapsack_2_typed knapsack_3_memoryview

all: $(TARGET)

%: %.py
	cython --embed -a $<
	g++ -O2 -o $@ $@.c `python2-config --cflags --ldflags`


.PHONY: clean
clean:
	rm -f *.c $(TARGET) *.html