CC := nvcc
CFLAGS := -g -dc
LDFLAGS := -lm

all: password-cracker-gpu

clean:
	rm -f password-cracker-gpu

password-cracker-gpu: password-cracker.cu gpu_md5.cu gpu_md5.cuh
	$(CC) $(CFLAGS) -o password-cracker-gpu password-cracker.cu gpu_md5.cu $(LDFLAGS)

# password-cracker: password-cracker.cu
#	$(CC) $(CFLAGS) -o password-cracker password-cracker.cu $(LDFLAGS)