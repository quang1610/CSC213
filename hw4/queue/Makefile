CC := clang
CFLAGS := -g -Wall -Werror

all: queue

clean:
	rm -rf queue queue.dSYM

queue: queue.c
	$(CC) $(CFLAGS) -o queue queue.c
