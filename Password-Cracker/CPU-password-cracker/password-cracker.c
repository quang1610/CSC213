/// @author Quang Nguyen nguyenqu2
#define _GNU_SOURCE

#include <openssl/md5.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#define MAX_USERNAME_LENGTH 64
#define PASSWORD_LENGTH 6
#define PASSWORD_SPACE_SIZE 308915776
#define NUM_THREADS 4
#define CHAR_NUM 26
#define CRACKED 1
#define NOT_CRACKED 0

/************** SUPPORT STRUCTURE *************************/
/**
 * This is a node in password_set_t that contains information about a user and their password.
 */
typedef struct user_password {
    char username[MAX_USERNAME_LENGTH];
    uint8_t password_hash[MD5_DIGEST_LENGTH];
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

/**
 * Struct that pack the input parameters for crack_password_list_worker function
 */
typedef  struct password_cracker_args {
    password_set_t *passwords;
    int job_index;
}password_cracker_args_t;

/**************** GLOBAL VARIABLE *************************/
int cracked_password_num = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/************************* Part A *************************/
/**
 * Find a six character lower-case alphabetic password that hashes
 * to the given hash value. Complete this function for part A of the lab.
 *
 * \param input_hash  An array of MD5_DIGEST_LENGTH bytes that holds the hash of a password
 * \param output      A pointer to memory with space for a six character password + '\0'
 * \returns           0 if the password was cracked. -1 otherwise.
 */
int crack_single_password(uint8_t *input_hash, char *output) {
    // Take our candidate password and hash it using MD5
    char *candidate_password = malloc(sizeof(char) * (PASSWORD_LENGTH + 1));
    strcpy(candidate_password, "aaaaaa");
    for (int i = 0; i < PASSWORD_SPACE_SIZE; i ++) {
        // generate candidate password
        int div = i;
        for (int j = PASSWORD_LENGTH - 1; j >= 0; j--) {
            candidate_password[j] = (char) ('a' + div % CHAR_NUM);
            div = div / CHAR_NUM;
        }
        // checking password hash
        uint8_t candidate_hash[MD5_DIGEST_LENGTH]; //< This will hold the hash of the candidate password

        MD5((unsigned char *) candidate_password, strlen(candidate_password),
            candidate_hash); //< Do the hash

        // Now check if the hash of the candidate password matches the input hash
        if (memcmp(input_hash, candidate_hash, MD5_DIGEST_LENGTH) == 0) {
            // Match! Copy the password to the output and return 0 (success)
            memcpy(output, candidate_password, sizeof(char) * (PASSWORD_LENGTH + 1));
            free(candidate_password);
            return 0;
        }
    }
    free(candidate_password);
    return -1;
}

/********************* Parts B & C ************************/
/**
 * Initialize a password set.
 * Complete this implementation for part B of the lab.
 *
 * \param passwords  A pointer to allocated memory that will hold a password set
 */
void init_password_set(password_set_t *passwords) {
    passwords->head = NULL;
    passwords->user_num = 0;
}

/**
 * Add a password to a password set
 * Complete this implementation for part B of the lab.
 *
 * \param passwords   A pointer to a password set initialized with the function above.
 * \param username    The name of the user being added. The memory that holds this string's
 *                    characters will be reused, so if you keep a copy you must duplicate the
 *                    string. I recommend calling strdup().
 * \param password_hash   An array of MD5_DIGEST_LENGTH bytes that holds the hash of this user's
 *                        password. The memory that holds this array will be reused, so you must
 *                        make a copy of this value if you retain it in your data structure.
 */
void add_password(password_set_t *passwords, char *username, uint8_t *password_hash) {
    /// init new user
    user_password_t *new_user = malloc(sizeof(user_password_t));
    memcpy(new_user->username, username, MAX_USERNAME_LENGTH);
    memcpy(new_user->password_hash, password_hash, sizeof(uint8_t) * MD5_DIGEST_LENGTH);
    new_user->cracked_password = NOT_CRACKED;

    /// add new user
    new_user->next = passwords->head;
    passwords->head = new_user;
    passwords->user_num ++;
}
/**
 * Cracking list password worker. These function will be called by a thread, with expected input to be a struct contains
 * list of the users' password to be cracked and a job_index, indicating which part of candidate passwords list it is
 * going to brute force.
 *
 * Return NULL
 */
void* crack_password_list_worker (void * args) {
    // get the parameters
    password_set_t *passwords = ((password_cracker_args_t *) args)->passwords;
    int job_index = ((password_cracker_args_t *) args)->job_index;

    // set up candidate password
    char *candidate_password = malloc(sizeof(char) * (PASSWORD_LENGTH + 1));
    strcpy(candidate_password, "aaaaaa");
    // !this works since password space size % num threads = 0
    int start_candidate_password_index = PASSWORD_SPACE_SIZE / NUM_THREADS * job_index;
    int end_candidate_password_index = PASSWORD_SPACE_SIZE / NUM_THREADS * (job_index + 1);
    for (int i = start_candidate_password_index; i < end_candidate_password_index; i ++) {
        // generate candidate password
        int div = i;
        for (int j = 5; j >= 0; j--) {
            candidate_password[j] = (char) ('a' + div % CHAR_NUM);
            div = div / CHAR_NUM;
        }
        // checking password hash
        uint8_t candidate_hash[MD5_DIGEST_LENGTH]; //< This will hold the hash of the candidate password
        MD5((unsigned char *) candidate_password, strlen(candidate_password),
            candidate_hash); //< Do the hash

        // Now check if the hash of the candidate password matches the input hash
        user_password_t *cursor = passwords->head;
        while (cursor != NULL) {
            if (cursor->cracked_password == CRACKED) {
                // if this user has already been cracked, we don't need to check it
                cursor = cursor->next;
                continue;
            }
            if (memcmp(cursor->password_hash, candidate_hash, MD5_DIGEST_LENGTH) == 0) {
                // Match! print the candidate password then continue with new candidate
                pthread_mutex_lock(&lock);
                cursor->cracked_password = CRACKED;
                cracked_password_num += 1;
                pthread_mutex_unlock(&lock);
                printf("%s %s\n", cursor->username, candidate_password);
                break;
            }
            if (cracked_password_num == passwords->user_num) {
                // return early as we cracked all the passwords
                return  NULL;
            }
            cursor = cursor->next;
        }
    }
    return NULL;
}


/**
 * Crack all of the passwords in a set of passwords. The function should print the username
 * and cracked password for each user listed in passwords, separated by a space character.
 * Complete this implementation for part B of the lab.
 * \param   passwords       This is the list of user passwords that we need to crack
 *          thread_index    This will decide what part of the candidate passwords list will this worker solve,
 *                          ranging from 0 to 3
 * \returns The number of passwords cracked in the list
 */
int crack_password_list(password_set_t *passwords) {
    pthread_t threads[NUM_THREADS];
    password_cracker_args_t args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].passwords = passwords;
        args[i].job_index = i;
        pthread_create(&(threads[i]), NULL, crack_password_list_worker, &(args[i]));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    return cracked_password_num;
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
int md5_string_to_bytes(const char *md5_string, uint8_t *bytes) {
    // Check for a valid MD5 string
    if (strlen(md5_string) != 2 * MD5_DIGEST_LENGTH) return -1;

    // Start our "cursor" at the start of the string
    const char *pos = md5_string;

    // Loop until we've read enough bytes
    for (size_t i = 0; i < MD5_DIGEST_LENGTH; i++) {
        // Read one byte (two characters)
        int rc = sscanf(pos, "%2hhx", &bytes[i]);
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
        // The input MD5 hash is a string in hexadecimal. Convert it to bytes.
        uint8_t input_hash[MD5_DIGEST_LENGTH];
        if (md5_string_to_bytes(argv[2], input_hash)) {
            fprintf(stderr, "Input has value %s is not a valid MD5 hash.\n", argv[2]);
            exit(1);
        }

        // Now call the crack_single_password function
        char result[PASSWORD_LENGTH + 1];
        if (crack_single_password(input_hash, result)) {
            printf("No matching password found.\n");
        } else {
            printf("%s\n", result);
        }

    } else if (strcmp(argv[1], "list") == 0) {
        // Make and initialize a password set
        password_set_t passwords;
        init_password_set(&passwords);

        // Open the password file
        FILE *password_file = fopen(argv[2], "r");
        if (password_file == NULL) {
            perror("opening password file");
            exit(2);
        }

        int password_count = 0;

        // Read until we hit the end of the file
        while (!feof(password_file)) {
            // Make space to hold the username
            char username[MAX_USERNAME_LENGTH];

            // Make space to hold the MD5 string
            char md5_string[MD5_DIGEST_LENGTH * 2 + 1];

            // Make space to hold the MD5 bytes
            uint8_t password_hash[MD5_DIGEST_LENGTH];

            // Try to read. The space in the format string is required to eat the newline
            if (fscanf(password_file, "%s %s ", username, md5_string) != 2) {
                fprintf(stderr, "Error reading password file: malformed line\n");
                exit(2);
            }

            // Convert the MD5 string to MD5 bytes in our new node
            if (md5_string_to_bytes(md5_string, password_hash) != 0) {
                fprintf(stderr, "Error reading MD5\n");
                exit(2);
            }

            // Add the password to the password set
            add_password(&passwords, username, password_hash);
            password_count++;
        }

        // Now run the password list cracker
        int cracked = crack_password_list(&passwords);

        printf("Cracked %d of %d passwords.\n", cracked, password_count);

        // free passwords set
        user_password_t *cursor = passwords.head;
        user_password_t *temp = NULL;
        while (cursor != NULL) {
            temp = cursor;
            cursor = cursor->next;
            free(temp);
        }

    } else {
        print_usage(argv[0]);
        exit(1);
    }
    return 0;
}
