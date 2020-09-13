CC := nvcc
CFLAGS := -g -dc
LDFLAGS := -lm

all: password-cracker-gpu generate_md5_hash

clean:
	rm -f password-cracker-gpu
	rm -f generate_md5_hash

password-cracker-gpu:
	$(CC) $(CFLAGS) -o password-cracker-gpu password-cracker.cu $(LDFLAGS)

generate_md5_hash:
	$(CC) $(CFLAGS) -o generate_md5_hash generate_md5_hash.cu $(LDFLAGS)