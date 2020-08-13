//
// Created by Nguyễn Đức Quang on 5/7/20.
//
#include <string.h>
#include <stdlib.h>

#include "p2p_thread.h"
#include "socket.h"
#include "time.h"
#include "ui.h"

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

    /// local pthread_set
    pthread_set_t set;
    pthread_set_init(&set);

    /// local argument set for the threads
    mess_listening_args_t args_set;
    mess_listening_args_init(&args_set);

    /// waiting for the message, reset timeout everytime we read new message.
    message_t new_message;
    while (*listening_args->terminate == FALSE) {
        if (read_message(&new_message, from_fd) == sizeof(message_t)) {
            /// check if we have new peer request, which we need to connect to them and listen to them
            if (new_message.type == TYPE_ADD_PEER) {
                /// check if this is an old message first!
                if (mess_record_add(record, &new_message) == MESS_RECORD_ADD_SUCCESS) {
                    /// this is a new peer request
                    /// first we try to broadcast it to our peers
                    for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
                        pthread_mutex_lock(&list->mutex[i]);
                        peer_t *cursor = list->table[i];

                        while (cursor != NULL) {
                            send_message(&new_message, cursor->to_peer);
                            cursor = cursor->next;
                        }
                        pthread_mutex_unlock(&list->mutex[i]);
                    }

                    /// then try adding new peer if I haven't
                    int socket_fd = socket_connect(new_message.sender_server_name,
                                                   new_message.sender_port);  // this is different from from_fd.
                    if (socket_fd == -1)
                        continue;
                    if (peer_list_add_peer(list, socket_fd, new_message.sender_username) != PEER_LIST_ADD_SUCCESSFUL) {
                        /// already added as a peer
                        close(socket_fd);
                    } else {
                        /// NEW PEER, let's listen to him!
                        pthread_t *new_thread = (pthread_t *) malloc(sizeof(pthread_t));
                        pthread_set_add(&set, new_thread);

                        /// set up args for new_thread
                        message_listening_t *new_args = (message_listening_t *) malloc(sizeof(message_listening_t));
                        new_args->terminate = listening_args->terminate;
                        new_args->from_fd = socket_fd;
                        new_args->record = record;
                        new_args->list = list;
                        mess_listening_args_add(&args_set, new_args);

                        pthread_create(new_thread, NULL, listening_to_peer_worker, new_args);
                    }

                    /// sending you username back to sender, we would hope that sender could read our name and add it to his peer_list.
                    message_t *my_name_message = message_generate(TYPE_MY_NAME, list->my_username, NULL, NULL, -1);
                    send_message(my_name_message, peer_list_find_peer_fd(list, new_message.sender_username));

                    /// print out the introduction message
                    ui_display(new_message.sender_username, "Enter the room!\n");
                }
            } else {
                /// This will skip the TYPE_ADD_PEER because we just process it
                process_message(&new_message, record, list, from_fd);
            }
        }
    }

    pthread_set_destroy(&set);
    mess_listening_args_destroy(&args_set);

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
            new_listening_args->record = record;
            new_listening_args->list = list;
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
