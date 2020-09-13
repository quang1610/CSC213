CC := nvcc
CFLAGS := -g -dc
LDFLAGS := -lm

all: password-cracker-gpu

clean:
	rm -f password-cracker-gpu

password-cracker-gpu:
	$(CC) $(CFLAGS) -o password-cracker-gpu password-cracker.cu $(LDFLAGS)

# password-cracker: password-cracker.cu
#	$(CC) $(CFLAGS) -o password-cracker password-cracker.cu $(LDFLAGS)