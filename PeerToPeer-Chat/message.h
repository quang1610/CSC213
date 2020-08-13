//
// Created by Nguyễn Đức Quang on 5/6/20.
//

#include <unistd.h>
#include <stdio.h>
#include <pthread.h>

#include "utility.h"
#include "hashcode.h"
#include "peer_list.h"

#ifndef CSC213_MESSAGE_H
#define CSC213_MESSAGE_H

#endif //CSC213_MESSAGE_H

/// SUPPORT  STRUCT
/**
 * This is the structure of a message that is sent accross the network
 */
typedef struct message {
    int type;
    long time_stamp;
    unsigned long hash_code;

    char sender_username[USERNAME_LEN];
    char message_content[MESSAGE_LEN];

    char sender_server_name[SERVER_NAME_LEN];
    unsigned short sender_port;
} message_t;

/**
 * This is the structure that is used to identify if we get this message before
 */
typedef struct mess_record_block {
    long time_stamp;
    unsigned long hash_code;

    struct mess_record_block *next;
    struct mess_record_block *previous;
} mess_record_block_t;

/**
 * This is the structure that hold the record of the messages.
 */
typedef struct mess_record {
    char my_username[USERNAME_LEN];
    pthread_mutex_t mutex[MESS_RECORD_TABLE_SIZE];
    mess_record_block_t *table[MESS_RECORD_TABLE_SIZE];
    int size[MESS_RECORD_TABLE_SIZE];
} mess_record_t;

/// MESSAGE RECORD FUNCTIONS
/**
 * This function initialize the record_t struct. It would init mutex locks, set table entries to NULL, set size entries
 * to 0, and set record->my_username to input my_username.
 * @param record
 * @param my_username
 */
void mess_record_init(mess_record_t *record, const char *my_username);

/**
 * This function would destroy the record
 * @param record
 */
void mess_record_destroy(mess_record_t *record);

/**
 * This function would check if any record with correspond hashcode and time_stamp.
 * @param record
 * @param hashcode
 * @param time_stamp
 * @return
 */
int mess_record_check_present(mess_record_t *record, unsigned long hashcode, long time_stamp);

/**
 * This function would try to add message->hashcode and message->time_stamp to the record. It would not add if this
 * message is sent from same user or if the message is already added.
 * @param record
 * @param message
 * @return
 */
int mess_record_add(mess_record_t *record, message_t *message);

/**
 * This function would check any buckets has more than NUM_OLD_MESSAGES_ALLOW. If any surpass that limit, this function
 * would free those old record until we reach half of the limit
 * @param record
 */
void mess_record_clean_up(mess_record_t *record);

/// MESSAGE FUNCTIONS
/**
 * This function would generate and return a message from the input.
 * @param type
 * @param sender_username
 * @param message_content
 * @param sender_server_name
 * @param port
 * @return
 */
message_t *
message_generate(int type, const char *sender_username, const char *message_content, const char *sender_server_name,
                 unsigned short port);

/**
 * This function would return the type of the message
 * @param message
 * @return
 */
int message_type(message_t *message);

/**
 * This function would send message to file_descriptor fd
 * @param message
 * @param fd
 * @return
 */
ssize_t send_message(message_t *message, int fd);

void send_all(message_t *message, peer_list_t *list);

/**
 * This function would read message from file_descriptor fd
 * @param message
 * @param fd
 * @return
 */
ssize_t read_message(message_t *message, int fd);

/**
 * This function would process the message based on its type.
 * If the message is normal type (ie the "text message" that needed to be displayed, it add message to record, display it,
 * and broadcast it to peers in peer_list.
 *
 * If the message is peer request, we would broadcast it to our peers first, then we try to add it if possible.
 *
 * If the message is peer remove request, we would broadcast it to our peers first, then we try to remove it if possible.
 * @param message
 * @param record
 * @param list
 * @return
 */
int process_message(message_t *message, mess_record_t *record, peer_list_t *list, int from_fd);
