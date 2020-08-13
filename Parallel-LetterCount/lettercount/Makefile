CC := clang
CFLAGS := -g -Wall -Werror

all: lettercount

clean:
	rm -rf lettercount lettercount.dSYM

lettercount: lettercount.c
	$(CC) $(CFLAGS) -o lettercount lettercount.c -lpthread
