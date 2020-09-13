#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu_md5.cu"

#define PASSWORD_LENGTH 6
#define MD5_UNSIGNED_HASH_LEN 4

// This program take a string and return a hashcode for it. 

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("The program require an argument, a string, to work!\n");
        exit(1);
    }

    if (strlen(argv[1]) != PASSWORD_LENGTH) {
        printf("The password's length must be %d\n", PASSWORD_LENGTH);
        exit(1);
    }

    unsigned *hash_code = (unsigned*) malloc(sizeof(unsigned) * MD5_UNSIGNED_HASH_LEN);
    md5((unsigned char*) argv[1], PASSWORD_LENGTH, hash_code);

    for (int i = 0; i < 4; i++) {
        printf("%u", hash_code[i]);
    }
    printf("\n");

    free(hash_code);

    return 0;
}