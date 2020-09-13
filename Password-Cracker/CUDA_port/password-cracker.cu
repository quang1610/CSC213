/// @author Quang Nguyen nguyenqu2
//#define _GNU_SOURCE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#include "gpu_md5.cu"

#define MAX_USERNAME_LENGTH 64
#define PASSWORD_LENGTH 6
#define PASSWORD_SPACE_SIZE 308915776
#define NUM_THREADS 512
#define CHAR_NUM 26
#define CRACKED 1
#define NOT_CRACKED -1


/************** SUPPORT STRUCTURE *************************/
/**
 * This is a node in password_set_t that contains information about a user and their password.
 */
typedef struct user_password {
    char username[MAX_USERNAME_LENGTH];
    unsigned password_hash[4];
    int cracked_password;

    struct user_password *next;
} user_password_t;

/**
 * This struct is the root of the data structure that will hold users and hashed passwords.
 * This could be any type of data structure you choose: list, array, tree, hash table, etc.
 * Implement this data structure for part B of the lab.
 */
typedef struct password_set {
    user_password_t *head;
    int user_num;
} password_set_t;

/******************* Device code **************************/

__global__ void single_crack_MD5(uint8_t *input_hash, char* output, int *cracked, int id_offset) {
    if (*cracked == NOT_CRACKED) {
        int N = threadIdx.x + blockIdx.x * blockDim.x + id_offset;
        if (N >= PASSWORD_SPACE_SIZE) {
            return;
        } 

        // generate candidate based on N
        char *candidate_password = (char *) malloc(sizeof(char) * (PASSWORD_LENGTH + 1));
        char temp[] = "aaaaaa";
        memcpy(candidate_password, temp, sizeof(char) * (PASSWORD_LENGTH + 1));
        for (int j = PASSWORD_LENGTH - 1; j >= 0; j--) {
            candidate_password[j] = (char) ('a' + N % CHAR_NUM);
            N = N / CHAR_NUM;
        }

        // generate candidate hash
        uint8_t *candidate_hash = (uint8_t*) malloc(sizeof(uint8_t) * MD5_UNSIGNED_HASH_LEN);
        md5((unsigned char*) candidate_password, PASSWORD_LENGTH, candidate_hash);

        // compare candidate hash with input hash
        for (int i = 0; i < MD5_UNSIGNED_HASH_LEN; i++) {
            if (input_hash[i] != candidate_hash[i]) {
                free(candidate_password);
                free(candidate_hash);
                return;
            }
        }
        
        // update cracked
        atomicAdd(cracked, 100 + 1);
        memcpy(output, candidate_password, sizeof(char) * (PASSWORD_LENGTH + 1));

        free(candidate_password);
        free(candidate_hash);
    }
}


/******************** Password crack code *****************/
void crack_single_password(uint8_t *input_hash, char *output, int *cracked) {
    int num_block = 1000;
    int block_size = 500;

    int total_thread = 0;

    while (total_thread < PASSWORD_SPACE_SIZE) {
        if (*cracked == NOT_CRACKED) {
            single_crack_MD5<<<num_block, block_size>>>(input_hash, output, cracked, total_thread);
            cudaDeviceSynchronize();

            total_thread += num_block * block_size;
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
    if (argc != 3) {
        print_usage(argv[0]);
        exit(1);
    }

    if (strcmp(argv[1], "single") == 0) {
        uint8_t *input_hash;
        cudaMallocManaged(&input_hash, sizeof(uint8_t) * MD5_UNSIGNED_HASH_LEN);

        int *cracked;
        cudaMallocManaged(&cracked, sizeof(int));
        *cracked = NOT_CRACKED;

        // The input MD5 hash is a string in hexadecimal. Convert it to bytes.
        if (md5_string_to_bytes(argv[2], input_hash)) {
            fprintf(stderr, "Input has value %s is not a valid MD5 hash.\n", argv[2]);

            // Free variable
            cudaFree(input_hash);
            cudaFree(cracked);
            exit(1);
        }

        // Now call the crack_single_password function
        char *result;
        cudaMallocManaged(&result, sizeof(char) * (PASSWORD_LENGTH + 1));

        crack_single_password (input_hash, result, cracked);
        if (*cracked == NOT_CRACKED) {
            printf("No matching password found.\n");
        } else {
            printf("%s\n", result);
        }

        // Free variable
        cudaFree(input_hash);
        cudaFree(cracked);

    } else if (strcmp(argv[1], "list") == 0) {
    //     // Make and initialize a password set
    //     password_set_t passwords;
    //     init_password_set(&passwords);

    //     // Open the password file
    //     FILE *password_file = fopen(argv[2], "r");
    //     if (password_file == NULL) {
    //         perror("opening password file");
    //         exit(2);
    //     }

    //     int password_count = 0;

    //     // Read until we hit the end of the file
    //     while (!feof(password_file)) {
    //         // Make space to hold the username
    //         char username[MAX_USERNAME_LENGTH];

    //         // Make space to hold the MD5 string
    //         char md5_string[MD5_UNSIGNED_HASH_LEN * 4 + 1];

    //         // Make space to hold the MD5 bytes
    //         unsigned password_hash[MD5_UNSIGNED_HASH_LEN];

    //         // Try to read. The space in the format string is required to eat the newline
    //         if (fscanf(password_file, "%s %s ", username, md5_string) != 2) {
    //             fprintf(stderr, "Error reading password file: malformed line\n");
    //             exit(2);
    //         }

    //         // Convert the MD5 string to MD5 bytes in our new node
    //         if (md5_string_to_unsigned(md5_string, password_hash) != 0) {
    //             fprintf(stderr, "Error reading MD5\n");
    //             exit(2);
    //         }

    //         // Add the password to the password set
    //         add_password(&passwords, username, password_hash);
    //         password_count++;
    //     }


    //     // Now run the password list cracker
    //     int cracked = crack_password_list(&passwords);

    //     printf("Cracked %d of %d passwords.\n", cracked, password_count);

    //     // free passwords set
    //     user_password_t *cursor = passwords.head;
    //     user_password_t *temp = NULL;
    //     while (cursor != NULL) {
    //         temp = cursor;
    //         cursor = cursor->next;
    //         free(temp);
    //     }

    // } else {
    //     print_usage(argv[0]);
    //     exit(1);
    }
    return 0;
}