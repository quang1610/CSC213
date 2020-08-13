
all: malloc-test

clean:
	rm -rf malloc-test

malloc-test: malloc-test.c
	clang -fno-omit-frame-pointer -o malloc-test malloc-test.c -D_GNU_SOURCE
