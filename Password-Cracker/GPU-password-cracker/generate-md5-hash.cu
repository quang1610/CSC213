#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu-md5.cu"

#define PASSWORD_LENGTH 6

/**
 * Convert a string representation of an MD5 hash to a sequence
 * of bytes. The input md5_string must be 32 characters long, and
 * the output buffer bytes must have room for MD5_DIGEST_LENGTH
 * bytes.
 *
 * \param md5_string  The md5 string representation
 * \param bytes       The destination buffer for the converted md5 hash
 * \returns           0 on success, -1 otherwise
 */
int md5_string_to_bytes(const char *md5_string, uint8_t *hash_code) {
    // Check for a valid MD5 string
    if (strlen(md5_string) != 2 * MD5_UNSIGNED_HASH_LEN) return -1;

    // Start our "cursor" at the start of the string
    const char *pos = md5_string;

    // Loop until we've read enough bytes
    for (size_t i = 0; i < MD5_UNSIGNED_HASH_LEN; i++) {
        // Read one byte (two characters)
        int rc = sscanf(pos, "%2hhx", &hash_code[i]);
        if (rc != 1) return -1;

        // Move the "cursor" to the next hexadecimal byte
        pos += 2;
    }
    return 0;
}

typedef union hcu {
    unsigned char b[MD5_UNSIGNED_HASH_LEN];
} HCunion;

// This program take a string and return a hashcode for it. 

__global__ void get_md5_hashcode(unsigned char *password, int password_len, uint8_t *hash_code) {
    md5((unsigned char*) password, PASSWORD_LENGTH, hash_code);
}

int main(int argc, char **argv) {
    // check for valid arguments
    if (argc != 2) {
        printf("The program require an argument, a string, to work!\n");
        exit(1);
    }

    if (strlen(argv[1]) != PASSWORD_LENGTH) {
        printf("The password's length must be %d\n", PASSWORD_LENGTH);
        exit(1);
    }

    // allocate memory for hashcode and password to be used in GPU mem
    uint8_t *hash_code;
    cudaMallocManaged(&hash_code, sizeof(uint8_t) * (MD5_UNSIGNED_HASH_LEN));

    char *gpu_password;
    cudaMalloc(&gpu_password, sizeof(char) * (PASSWORD_LENGTH + 1));
    cudaMemcpy(gpu_password, argv[1], sizeof(char) * (PASSWORD_LENGTH + 1), cudaMemcpyHostToDevice);

    // set the hashcode into hash_code string.
    get_md5_hashcode<<<1,1>>>((unsigned char *) gpu_password, PASSWORD_LENGTH, hash_code);
    cudaDeviceSynchronize();

    // print the pass code in hex form
    HCunion h;
    memcpy(&h, hash_code, sizeof(unsigned char) * MD5_UNSIGNED_HASH_LEN);
    char *temp_str = malloc(sizeof(char) * 33);
    for (int i = 0; i < MD5_UNSIGNED_HASH_LEN; i++) {
        sprintf(&(temp_str[i*2]), "%02x", h.b[i]);
    }
    printf("%s\n", temp_str);

    cudaFree(hash_code);
    cudaFree(gpu_password);

    return 0;
}