#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define USERNAME_LEN 256
#define MESSAGE_LEN 500
#define SERVER_NAME_LEN 256

#define MESS_RECORD_TABLE_SIZE 100
#define NEW_MESSAGE -1
#define OLD_MESSAGE 1
#define MESS_RECORD_ADD_SUCCESS 1
#define MESS_RECORD_ADD_FAIL -1
#define MESS_RECORD_DUPLICATE_ADD -2
#define NUM_OLD_MESSAGES_ALLOW 20
#define MESS_PROCESS_SUCCESSFUL 1
#define MESS_PROCESS_FAIL -1

#define PEER_LIST_TABLE_SIZE 100
#define IN_PEER_LIST 1
#define NOT_IN_PEER_LIST -1
#define PEER_LIST_ADD_SUCCESSFUL 1
#define PEER_LIST_ADD_FAIL -1
#define PEER_LIST_DUPLICATE_ADD -2

#define TYPE_NORMAL 1
#define TYPE_ADD_PEER 2
#define TYPE_REMOVE_PEER 3
#define TYPE_MY_NAME 4

#define MAX_QUEUE_CONNECTION 20
#define LISTENING_TIME_OUT 3600
#define PTHREAD_SET_INIT_CAP 4
#define FALSE 0

typedef struct message {
    int type;
    long time_stamp;
    unsigned long hash_code;

    char sender_username[USERNAME_LEN];
    char message_content[MESSAGE_LEN];

    char sender_server_name[SERVER_NAME_LEN];
    unsigned short sender_port;
} message_t;

unsigned long hashcode(const char *str) {
    unsigned long hash = 5381;
    int len = (int) strlen(str);

    for (int i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + str[i]; /* hash * 33 + c */
    }

    return hash;
}

message_t *
message_generate(int type, const char *sender_username, const char *message_content, const char *sender_server_name,
                 unsigned short port) {
    message_t *message = (message_t*) malloc(sizeof(message_t));

    /// set type
    message->type = type;

    /// set time stamp
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    message->time_stamp = ((long)time.tv_sec) * 1000000000  + time.tv_nsec;

    /// set hashcode
    char buffer[USERNAME_LEN + 50];
    sprintf(buffer, "%ld", message->time_stamp);
    printf("%s\n", buffer);
    strcat(buffer, sender_username);
    message->hash_code = hashcode(buffer);

    /// set sender's username
    strcpy(&message->sender_username[0], sender_username);

    /// set message's content
    strcpy(&message->message_content[0], message_content);

    /// set sender's servername/IP address
    strcpy(&message->sender_server_name[0], sender_server_name);

    /// set sender's server's port
    message->sender_port = port;

    return message;
}

typedef struct name {
    char name[USERNAME_LEN];
} name_t;

int main() {
    char username[USERNAME_LEN];
    char mess[MESSAGE_LEN];
    char servername[SERVER_NAME_LEN];

    strcpy(username, "student1");
    strcpy(servername, "localhost");
    strcpy(mess, "Hi there!");

    name_t new_name = {"quang"};

    message_t *message = message_generate(TYPE_NORMAL, new_name.name, &mess[0], &servername[0], 49560);
    printf("%s\n", message->sender_username);
    printf("%s\n", message->sender_server_name);
    printf("%s\n", message->message_content);

    free(message);
    return 0;
}

