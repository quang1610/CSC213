CC := nvcc
CFLAGS := -g $(shell sdl2-config --cflags)
LDFLAGS := $(shell sdl2-config --libs) -lm

all: password-cracker

clean:
	rm -f password-cracker

gpu_md5.o:
	$(CC) $(CFLAGS) gpu_md5.cu $(LDFLAGS)

# password-cracker: password-cracker.cu
#	$(CC) $(CFLAGS) -o password-cracker password-cracker.cu $(LDFLAGS)