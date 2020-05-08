//
// Created by Nguyễn Đức Quang on 5/7/20.
//
#include <string.h>
#include <stdlib.h>

#include "p2p_thread.h"
#include "socket.h"
#include "time.h"

/// MESSAGE_LISTENING_ARGS
void mess_listening_args_init(mess_listening_args_t *set) {
    set->args_set = malloc(sizeof(message_listening_t *) * PTHREAD_SET_INIT_CAP);
    set->capacity = PTHREAD_SET_INIT_CAP;
    set->args_counts = 0;
}

void mess_listening_args_destroy(mess_listening_args_t *set) {
    for (int i = set->args_counts - 1; i >= 0; i--) {
        free(set->args_set[i]);
    }
    set->args_counts = 0;
    set->capacity = PTHREAD_SET_INIT_CAP;
}

void mess_listening_args_add(mess_listening_args_t *set, message_listening_t *new_arg) {
    if (set->args_counts >= set->capacity) {
        set->args_set = realloc(set->args_set, sizeof(message_listening_t) * set->capacity * 2);
        set->capacity *= 2;
    }
    set->args_set[set->args_counts] = new_arg;
    set->args_counts += 1;
}

/// PTHREAD_SET FUNCTIONS
void pthread_set_init(pthread_set_t *set) {
    set->set = malloc(sizeof(pthread_t *) * PTHREAD_SET_INIT_CAP);
    set->capacity = PTHREAD_SET_INIT_CAP;
    set->thread_count = 0;
}

void pthread_set_destroy(pthread_set_t *set) {
    for (int i = set->thread_count - 1; i >= 0; i--) {
        pthread_cancel(*set->set[i]);
        free(set->set + i);
    }

    set->thread_count = 0;
    set->capacity = PTHREAD_SET_INIT_CAP;
}

void pthread_set_add(pthread_set_t *set, pthread_t *new_thread) {
    if (set->thread_count >= set->capacity) {
        set->set = (realloc(set->set, sizeof(pthread_t *) * set->capacity * 2));
        set->capacity *= 2;
    }

    set->set[set->thread_count] = new_thread;
    set->thread_count += 1;
}


/// WORKER FUNCTIONS
void *listening_to_peer_worker(void *args) {
    /// read the arguments
    message_listening_t *listening_args = (message_listening_t *) args;
    int from_fd = listening_args->from_fd;
    peer_list_t *list = listening_args->list;
    mess_record_t *record = listening_args->record;

    /// waiting for the message, reset timeout everytime we read new message.
    message_t new_message;

    while (*listening_args->terminate == FALSE) {
        if (read_message(&new_message, from_fd) == sizeof(message_t)) {
            process_message(&new_message, record, list, from_fd);
        }
    }
    return NULL;
}

void *receiving_new_connection_worker(void *args) {
    reception_args_t *reception_args = (reception_args_t *) args;
    int my_server_socket_fd = reception_args->server_socket_fd;
    peer_list_t *list = reception_args->peer_list;
    mess_record_t *record = reception_args->mess_record;

    /// local pthread_set
    pthread_set_t set;
    pthread_set_init(&set);

    /// local argument set for the threads
    mess_listening_args_t args_set;
    mess_listening_args_init(&args_set);

    while (*reception_args->terminate == FALSE) {
        /// accepting new request!
        int client_socket = server_socket_accept(my_server_socket_fd);
        if (client_socket != -1) {
            pthread_t *new_thread = (pthread_t *) malloc(sizeof(pthread_t));
            /// added to pthread database
            pthread_set_add(&set, new_thread);

            /// create argument for the threads
            message_listening_t *new_listening_args = (message_listening_t *) malloc(sizeof(message_listening_t));
            new_listening_args->terminate = reception_args->terminate;
            new_listening_args->from_fd = client_socket;
            new_listening_args->record = reception_args->mess_record;
            new_listening_args->list = reception_args->peer_list;
            /// added to listening arg database
            mess_listening_args_add(&args_set, new_listening_args);

            // start the thread
            pthread_create(new_thread, NULL, listening_to_peer_worker, (void *) new_listening_args);
        }
    }

    mess_listening_args_destroy(&args_set);
    pthread_set_destroy(&set);
    return NULL;
}
