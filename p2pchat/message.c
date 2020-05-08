//
// Created by Nguyễn Đức Quang on 5/6/20.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "message.h"
#include "socket.h"
#include "ui.h"

/// MESSAGE RECORD FUNCTIONS
void mess_record_init(mess_record_t *record, const char *my_username) {
    strcpy(&record->my_username[0], my_username);
    printf("set up record %s", record->my_username);
    for (int i = 0; i < MESS_RECORD_TABLE_SIZE; i++) {
        pthread_mutex_init(&record->mutex[i], NULL);
        record->table[i] = NULL;
        record->size[i] = 0;
    }
}

void mess_record_destroy(mess_record_t *record) {
    for (int i = 0; i < MESS_RECORD_TABLE_SIZE; i++) {
        pthread_mutex_lock(&record->mutex[i]);
        mess_record_block_t *cursor = record->table[i];
        record->table[i] = NULL;
        record->size[i] = 0;
        pthread_mutex_unlock(&record->mutex[i]);

        while (cursor != NULL) {
            mess_record_block_t *temp = cursor;
            cursor = cursor->next;

            free(temp);
        }
    }
}

int mess_record_check_present(mess_record_t *record, unsigned long hashcode, long time_stamp) {
    int index = (int) (hashcode % MESS_RECORD_TABLE_SIZE);

    pthread_mutex_lock(&record->mutex[index]);

    mess_record_block_t *cursor = record->table[index];

    while (cursor != NULL) {
        /// check if any message record block has same time stamp and same hashcode
        if (cursor->hash_code == hashcode && cursor->time_stamp == time_stamp) {
            pthread_mutex_unlock(&record->mutex[index]);
            return OLD_MESSAGE;
        }
        cursor = cursor->next;
    }

    pthread_mutex_unlock(&record->mutex[index]);
    return NEW_MESSAGE;
}

int mess_record_add(mess_record_t *record, message_t *message) {
    /// empty message
    if (message == NULL) {
        return MESS_RECORD_ADD_FAIL;
    }

    /// avoid sending message to yourself
    if (strcmp(record->my_username, message->sender_username) == 0) {
        return MESS_RECORD_DUPLICATE_ADD;
    }

    /// set up variable and find the right index
    unsigned hashcode = message->hash_code;
    long time_stamp = message->time_stamp;
    int index = (int) (hashcode % MESS_RECORD_TABLE_SIZE);

    pthread_mutex_lock(&record->mutex[index]);

    /// we need to check if the message record block has already been added ie this is an old message
    mess_record_block_t *cursor = record->table[index];
    while (cursor != NULL) {
        if (cursor->hash_code == hashcode && cursor->time_stamp == time_stamp) {
            pthread_mutex_unlock(&record->mutex[index]);
            return MESS_RECORD_DUPLICATE_ADD;
        }
        cursor = cursor->next;
    }

    /// create new message record block
    mess_record_block_t *new_mess_record_block = (mess_record_block_t*) malloc(sizeof(mess_record_block_t));
    if (new_mess_record_block == NULL) {
        pthread_mutex_unlock(&record->mutex[index]);
        return MESS_RECORD_ADD_FAIL;
    }

    new_mess_record_block->hash_code = hashcode;
    new_mess_record_block->time_stamp = time_stamp;
    new_mess_record_block->next = NULL;
    new_mess_record_block->previous = NULL;

    /// Adding to the table
    if (record->table[index] == NULL) {
        /// if the current table entry is empty
        record->table[index] = new_mess_record_block;
        record->size[index] += 1;
    } else {
        /// add new record block to the beginning of the bucket
        new_mess_record_block->next = record->table[index];
        record->table[index]->previous = new_mess_record_block;
        record->table[index] = new_mess_record_block;

        record->size[index] += 1;
    }

    pthread_mutex_unlock(&record->mutex[index]);
    return MESS_RECORD_ADD_SUCCESS;
}

void mess_record_clean_up(mess_record_t *record) {
    for (int i = 0; i < MESS_RECORD_TABLE_SIZE; i++) {
        pthread_mutex_lock(&record->mutex[i]);
        mess_record_block_t *cursor = record->table[i];

        /// if the number of old message accumulate more than the allowed quantity, we will have to "trim" it to only
        /// 1/2 of the allowed quantity
        if (record->size[i] >= NUM_OLD_MESSAGES_ALLOW) {
            int counter = 1;
            while (counter < NUM_OLD_MESSAGES_ALLOW / 2) {
                cursor = cursor->next;
                counter++;
            }
        }
        record->size[i] = NUM_OLD_MESSAGES_ALLOW / 2 - 1; /// we minus 1 because we free the current cursor
        pthread_mutex_unlock(&record->mutex[i]);

        /// cleaning
        while (cursor != NULL) {
            mess_record_block_t *temp = cursor;
            cursor = cursor->next;

            free(temp);
        }
    }
}

/// MESSAGE FUNCTIONS
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
    strcat(buffer, sender_username);
    message->hash_code = hashcode(buffer);

    /// set sender's username
    strcpy(&(message->sender_username[0]), sender_username);

    /// set message's content
    if (message_content == NULL) {
        message->message_content[0] = '\0';
    } else {
        strcpy(&(message->message_content[0]), message_content);
    }

    /// set sender's servername/IP address
    if (sender_server_name == NULL) {
        message->sender_server_name[0] = '\0';
    } else {
        strcpy(&(message->sender_server_name[0]), sender_server_name);
    }

    /// set sender's server's port
    message->sender_port = port;

    return message;
}

int message_type(message_t *message) {
    return message->type;
}

ssize_t send_message(message_t *message, int fd) {
    if(message == NULL || fd == -1) {
        perror("Cannot send message!\n");
        return -1;
    }
    ssize_t result = write(fd, message, sizeof(message_t));

    return result;
}

ssize_t read_message(message_t *message, int fd) {
    if (message == NULL || fd == -1) {
        perror("cannot read to a null message!\n");
        return 0;
    }
    ssize_t result = read(fd, message, sizeof(message_t));
    return result;
}

void send_all(message_t *message, peer_list_t *list) {
    for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
        pthread_mutex_lock(&list->mutex[i]);

        peer_t *cursor = list->table[i];
        while (cursor != NULL) {
            send_message(message, cursor->to_peer);
            cursor = cursor->next;
        }
        pthread_mutex_unlock(&list->mutex[i]);
    }
}

int process_message(message_t *message, mess_record_t *record, peer_list_t *list, int from_fd) {
    if (mess_record_add(record, message) == MESS_RECORD_ADD_SUCCESS) {
        if (message->type == TYPE_NORMAL) {
            /// this is a new message
            /// first we display the message
            ui_display(message->sender_username, message->message_content);

            /// then we broadcast it to our peer
            for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
                pthread_mutex_lock(&list->mutex[i]);
                peer_t *cursor = list->table[i];

                while (cursor != NULL) {
                    send_message(message, cursor->to_peer);
                    cursor = cursor->next;
                }
                pthread_mutex_unlock(&list->mutex[i]);
            }
        } else if (message->type == TYPE_ADD_PEER) {
            /// this is a new peer request
            /// first we try to broadcast it to our peers
            for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
                pthread_mutex_lock(&list->mutex[i]);
                peer_t *cursor = list->table[i];

                while(cursor != NULL) {
                    send_message(message, cursor->to_peer);
                    cursor = cursor->next;
                }
                pthread_mutex_unlock(&list->mutex[i]);
            }

            /// then try adding new peer if I haven't
            int socket_fd = socket_connect(message->sender_server_name, message->sender_port);  // this is different from from_fd.
            if (socket_fd == -1)
                return MESS_PROCESS_FAIL;
            if (peer_list_add_peer(list, socket_fd, message->sender_username) != PEER_LIST_ADD_SUCCESSFUL) {
                /// already added as a peer
                close(socket_fd);
            } else {
                /// NEW PEER, let's listen to him!
                pthread_t *new_thread = (pthread_t*)malloc(sizeof(pthread_t));

                /// set up args for new_thread
            }

            /// sending you username back to sender, we would hope that sender could read our name and add it to his peer_list.
            message_t *my_name_message = message_generate(TYPE_MY_NAME, list->my_username, NULL, NULL, -1);
            send_message(my_name_message, peer_list_find_peer_fd(list, message->sender_username));

            /// print out the introduction message
            ui_display(message->sender_username, "Enter the room!\n");

        } else if (message->type == TYPE_REMOVE_PEER) {
            /// this is a new peer removal request
            /// first we try to broadcast this message to our peer
            for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
                pthread_mutex_lock(&list->mutex[i]);
                peer_t *cursor = list->table[i];

                while(cursor != NULL) {
                    send_message(message, cursor->to_peer);
                    cursor = cursor->next;
                }
                pthread_mutex_unlock(&list->mutex[i]);
            }

            /// then we try to remove current peer
            peer_list_remove_peer(list, message->sender_username);

            /// notice that a person leave the room
            ui_display(message->sender_username, "Left room!\n");
        } else if (message->type == TYPE_MY_NAME) {
            /// this is a new my name message. For this type of message, we only need the sender username to add the from_fd
            /// this type of message is sent directly meaning from end to end, no middle man.
            if (peer_list_add_peer(list, from_fd, message->sender_username) == PEER_LIST_ADD_SUCCESSFUL) {
                ui_display(message->sender_username, "We are peer now!\n");
            }
        }
        return MESS_PROCESS_SUCCESSFUL;
    } else {
        return MESS_PROCESS_FAIL;
    }
}


