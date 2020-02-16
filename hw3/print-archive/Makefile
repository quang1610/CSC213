CC := clang
CFLAGS := -g -Wall -Werror

all: print-archive

clean:
	rm -rf print-archive print-archive.dSYM

print-archive: print-archive.c
	$(CC) $(CFLAGS) -o print-archive print-archive.c
