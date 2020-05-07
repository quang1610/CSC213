//
// Created by Nguyễn Đức Quang on 5/6/20.
//
#include <stdio.h>
#include <stdlib.h>

#include "message.h"

/// RECORD BLOCK FUNCTIONS


/// MESSAGE RECORD FUNCTIONS
void mess_record_init (mess_record_t *record) {
    for (int i = 0; i < MESS_RECORD_TABLE_SIZE; i++) {
        pthread_mutex_init(&record->mutex[i], NULL);
        record->table[i] = NULL;
        record->size[i] = 0;
    }
}

void mess_record_destroy (mess_record_t *record) {
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

int mess_record_check_present (mess_record_t *record, unsigned long hashcode, time_t time_stamp) {
    int index = (int) (hashcode % MESS_RECORD_TABLE_SIZE);

    pthread_mutex_lock(&record->mutex[index]);

    mess_record_block_t *cursor = record->table[index];

    while(cursor != NULL) {
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

int mess_record_add(mess_record_t *record, unsigned long hashcode, time_t time_stamp) {
    /// create new message record block
    mess_record_block_t *new_mess_record_block = malloc(sizeof(mess_record_block_t));
    if (new_mess_record_block == NULL)
        return MESS_RECORD_ADD_FAIL;

    new_mess_record_block->hash_code = hashcode;
    new_mess_record_block->time_stamp = time_stamp;

    /// add new message record block to the table
    int index = hashcode % MESS_RECORD_TABLE_SIZE;
    pthread_mutex_lock(&record->mutex[index]);

    if(record->table[index] == NULL) {
        /// if the current table entry is empty
        record->table[index] = new_mess_record_block;
        record->size[index] += 1;
        pthread_mutex_unlock(&record->mutex[index]);
    } else {
        mess_record_block_t *cursor = record->table[index];

        /// we need to check if the message record block has already been added ie this is an old message
        while (cursor != NULL) {
            if (cursor->hash_code == hashcode && cursor->time_stamp == time_stamp) {
                pthread_mutex_unlock(&record->mutex[index]);
                return MESS_RECORD_DUPLICATE_ADD;
            }
            cursor = cursor->next;
        }

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
        record->size[i] = NUM_OLD_MESSAGES_ALLOW/2 -1; /// we minus 1 because we free the current cursor
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
message_t *message_generate(int type, const char *sender_username, const char *message_content, const char *sender_server_name,
                            unsigned port);

int message_type(message_t *message);

int send_message(message_t *message, FILE *fd);

