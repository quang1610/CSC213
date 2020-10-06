/// @author Quang Nguyen nguyenqu2
//#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#include "gpu-md5.cu"

#define MAX_USERNAME_LENGTH 64
#define PASSWORD_LENGTH 6
#define PASSWORD_SPACE_SIZE 308915776
#define NUM_THREADS 512
#define CHAR_NUM 26
#define CRACKED 1
#define NOT_CRACKED -1

/******************* Device code **************************/
/**
 * This cuntion run on device to crack the code. The idea is that it generate a candidate password,
 * find its hashcode and compare it with input_hash. If we find out, we print the result to output,
 * set the cracked variable.
 * \param input_hash the given hash, belong to the password we need to crack.
 * \param output the correct password. We need to print value into this string
 * \param cracked the number to indicate whether we crack the code.
 * \param id_offset this is the number of passwords we check, act as an offset for N.
 *      this decides the candidate password.
 */
__global__ void single_crack_MD5(uint8_t *input_hash, char* output, int *cracked, int id_offset) {

    __shared__ uint8_t s_input_hash[MD5_UNSIGNED_HASH_LEN];

    // get N based on the number id of block. This is used to construct to candidate password.
    // N = 0 would give us "aaaaaa" 
    // N = 1 would give us "aaaaab" so on.

    int N = threadIdx.x + blockIdx.x * blockDim.x + id_offset;
    if (N >= PASSWORD_SPACE_SIZE) {
        return;
    } 

    // copy to input_hash to shared memory
    if (N - id_offset < MD5_UNSIGNED_HASH_LEN) {
        s_input_hash[N - id_offset] = input_hash[N - id_offset];
        if (N == 2109) {
            printf("copy %d\n", N - id_offset);
            printf("d %d\n", s_input_hash[N - id_offset]);
        }
    }
    __syncthreads();

    // generate candidate based on N
    char candidate_password[] = "aaaaaa";
    for (int j = PASSWORD_LENGTH - 1; j >= 0; j--) {
        candidate_password[j] = (char) ('a' + N % CHAR_NUM);
        N = N / CHAR_NUM;
    }

    if (N == 2109) {
        printf("yes!\n");
        printf("candidate %s\n", candidate_password);
    }
    
    // generate candidate hash
    uint8_t candidate_hash[MD5_UNSIGNED_HASH_LEN];
    md5((unsigned char*) &(candidate_password[0]), PASSWORD_LENGTH, &(candidate_hash[0]));

    // compare candidate hash with input hash
    for (int i = 0; i < MD5_UNSIGNED_HASH_LEN; i++) {
        if (s_input_hash[i] != candidate_hash[i]) {
            return;
        }
    }
        
    // update cracked
    *cracked = CRACKED;
    memcpy(output, &(candidate_password[0]), sizeof(char) * (PASSWORD_LENGTH + 1));
}


/******************** Password crack code *****************/
/**
 * This function call the gpu function to crack code. Each time, we test 1000 * 500 passwords until
 * we check all the password space.
 * \param input_hash the given hash, belong to the password we need to crack.
 * \param output the correct password. We need to print value into this string
 * \param cracked the number to indicate whether we crack the code.
 */
void crack_single_password(uint8_t *input_hash, char *output, int *cracked) {
    int num_block = 256;
    int block_size = 128;

    int tested_passwords = 0;

    // testing the each password
    while (tested_passwords < PASSWORD_SPACE_SIZE) {
        if (*cracked == NOT_CRACKED) {
            // call the gpu function
            single_crack_MD5<<<num_block, block_size>>>(input_hash, output, cracked, tested_passwords);
            cudaDeviceSynchronize();

            tested_passwords += num_block * block_size;
        } else {
            break;
        }
    }
}


/******************** Provided Code ***********************/
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

void print_usage(const char *exec_name) {
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s single <MD5 hash>\n", exec_name);
    fprintf(stderr, "  %s list <password file name>\n", exec_name);
}


int main(int argc, char **argv) {
    // check the input arguments' correctness
    if (argc != 2) {
        print_usage(argv[0]);
        exit(1);
    }

    // allocate variables to use on device and host
    uint8_t *input_hash;
    int *cracked;
    char *result;
    cudaMallocManaged(&input_hash, sizeof(uint8_t) * MD5_UNSIGNED_HASH_LEN);
    cudaMallocManaged(&cracked, sizeof(int));
    cudaMallocManaged(&result, sizeof(char) * (PASSWORD_LENGTH + 1));

    *cracked = NOT_CRACKED;

    // The input MD5 hash is a string in hexadecimal. Convert it to bytes.
    if (md5_string_to_bytes(argv[1], input_hash)) {
        fprintf(stderr, "Input has value %s is not a valid MD5 hash.\n", argv[1]);

        // Early exit. Free variable
        cudaFree(input_hash);
        cudaFree(cracked);
        cudaFree(result);
        exit(1);
    }

    // Now call the crack_single_password function
    // result hold the correct password.

    // call crack single password
    crack_single_password(input_hash, result, cracked);

    // check if we successfully cracked the password
    if (*cracked == NOT_CRACKED) {
        printf("No matching password found.\n");
    } else {
        printf("%s\n", result);
    }

    // Free variable
    cudaFree(result);
    cudaFree(input_hash);
    cudaFree(cracked);

    return 0;
}
