CC := clang
CFLAGS := -g -Wall -Werror

all: uniquelist-test

clean:
	rm -rf uniquelist-test uniquelist-test.dSYM

uniquelist-test: main.c uniquelist.c uniquelist.h
	$(CC) $(CFLAGS) -o uniquelist-test main.c uniquelist.c
