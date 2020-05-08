//
// Created by Nguyễn Đức Quang on 5/7/20.
//

#ifndef CSC213_P2P_THREAD_H
#define CSC213_P2P_THREAD_H

#endif //CSC213_P2P_THREAD_H

#include "message.h"

/// SUPPORT STRUCT
/// PTHREAD STRUCT

/// Thread management
typedef struct pthread_set {
    pthread_t **set;
    int capacity;
    int thread_count;
}pthread_set_t;

/// Reception work
typedef struct reception_args {
    int *terminate;
    int server_socket_fd;

    peer_list_t *peer_list;
    mess_record_t *mess_record;
} reception_args_t;

/// Message Listening job
typedef struct message_listening {
    int *terminate;
    int from_fd;

    mess_record_t *record;
    peer_list_t *list;
} message_listening_t;

typedef struct mess_listening_args {
    message_listening_t **args_set;
    int capacity;
    int args_counts;
} mess_listening_args_t;

/// MESSAGE_LISTENING_ARGS
void mess_listening_args_init(mess_listening_args_t *set);

void mess_listening_args_destroy(mess_listening_args_t *set);

void mess_listening_args_add(mess_listening_args_t *set, message_listening_t *new_arg);

/// PTHREAD_SET FUNCTIONs NOT CONCURRENCY
void pthread_set_init(pthread_set_t *set);

void pthread_set_destroy(pthread_set_t *set);

void pthread_set_add(pthread_set_t *set, pthread_t *new_thread);

/// WORKER FUNCTIONS
void *listening_to_peer_worker (void *args);

void *receiving_new_connection_worker (void *args);