CC := clang
CFLAGS := -g -Wall -Werror

all: libetsbefriends.so segfault-test lazycopy-test

clean:
	rm -rf libetsbefriends.so lazycopy-test segfault-test

libetsbefriends.so: libetsbefriends.c
	$(CC) -shared -fPIC -o libetsbefriends.so libetsbefriends.c

segfault-test: segfault-test.c libetsbefriends.so
	$(CC) -o segfault-test segfault-test.c -L. -letsbefriends

lazycopy-test: lazycopy-test.c lazycopy.c lazycopy.h
	$(CC) -o lazycopy-test lazycopy-test.c lazycopy.c
