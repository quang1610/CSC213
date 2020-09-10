CC := clang
CFLAGS := -g -Wall -Werror

# Special settings for macOS users. This assumes you installed openssl with the brew package manager
SYSTEM := $(shell uname -s)
ifeq ($(SYSTEM),Darwin)
	CFLAGS += -I./openssl@1.1/include -L./openssl@1.1/lib
else
	CFLAGS += -I./openssl/include -L./openssl/lib
endif

all: password-cracker

clean:
	rm -rf password-cracker password-cracker.dSYM

password-cracker: password-cracker.c
	$(CC) $(CFLAGS) -o password-cracker password-cracker.c -lcrypto -lpthread -lm
