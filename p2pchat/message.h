//
// Created by Nguyễn Đức Quang on 5/6/20.
//

#include <ntsid.h>
#include <stdio.h>
#include <pthread.h>

#include "utility.h"
#include "hashcode.h"
#include "peer_list.h"

#ifndef CSC213_MESSAGE_H
#define CSC213_MESSAGE_H

#endif //CSC213_MESSAGE_H

/// SUPPORT  STRUCT
typedef struct message {
    int type;
    time_t time_stamp;
    unsigned long hash_code;

    char sender_username[USERNAME_LEN];
    char message_content[MESSAGE_LEN];

    char sender_server_name[SERVER_NAME_LEN];
    unsigned sender_port;
} message_t;

typedef struct mess_record_block {
    time_t time_stamp;
    unsigned long hash_code;

    struct mess_record_block *next;
    struct mess_record_block *previous;
} mess_record_block_t;

typedef struct mess_record {
    pthread_mutex_t mutex[MESS_RECORD_TABLE_SIZE];
    mess_record_block_t *table[MESS_RECORD_TABLE_SIZE];
    int size[MESS_RECORD_TABLE_SIZE];
} mess_record_t;

/// RECORD BLOCK FUNCTIONS


/// MESSAGE RECORD FUNCTIONS
void mess_record_init (mess_record_t *record);

void mess_record_destroy (mess_record_t *record);

int mess_record_check_present (mess_record_t *record, unsigned long hashcode, time_t time_stamp);

int mess_record_add(mess_record_t *record, unsigned long hashcode, time_t time_stamp);

void mess_record_clean_up(mess_record_t *record);

/// MESSAGE FUNCTIONS
message_t *message_generate(int type, const char *sender_username, const char *message_content, const char *sender_server_name,
                            unsigned port);

int message_type(message_t *message);

int send_message(message_t *message, FILE *fd);

int process_message(message_t *message, mess_record_t *record, peer_list_t *list);

