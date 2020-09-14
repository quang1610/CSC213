CC := nvcc
CFLAGS :=
LDFLAGS := -lm

all: password-cracker-gpu generate-md5-hash

clean:
	rm -f password-cracker-gpu
	rm -f generate-md5-hash

password-cracker-gpu:
	$(CC) $(CFLAGS) -o password-cracker-gpu password-cracker.cu $(LDFLAGS)

generate_md5_hash:
	$(CC) $(CFLAGS) -o generate-md5-hash generate-md5-hash.cu $(LDFLAGS)