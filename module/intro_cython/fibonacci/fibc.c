#include "fibc.h"

int fibc(int n) {
    int i, tmp;
    int a=0, b=1;
    for (i = 0; i < n; ++i) {
        tmp = a;
        a = a + b;
        b = tmp;
    }
    return a;
}

int main (int argc, char *argv[])
{
    int result = 0;
    int i;
    for (i = 0; i < atoi(argv[2]); ++i)
      result = fibc(atoi(argv[1]));
    printf("%d\n", result);
}
