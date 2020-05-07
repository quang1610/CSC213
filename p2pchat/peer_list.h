//
// Created by Nguyễn Đức Quang on 5/6/20.
//

#ifndef CSC213_PEER_LIST_H
#define CSC213_PEER_LIST_H

#endif //CSC213_PEER_LIST_H

#include <stdio.h>
#include <pthread.h>

#include "utility.h"

/// SUPORT STRUCT

typedef struct peer {
    unsigned long hashcode; // created from username

    FILE *to_peer;
    FILE *from_peer;

    char peer_username[USERNAME_LEN];

    struct peer *next;
    struct peer *previous;
} peer_t;

typedef struct peer_list {
    pthread_mutex_t mutex[PEER_LIST_TABLE_SIZE];
    peer_t *table[PEER_LIST_TABLE_SIZE];
    int size[PEER_LIST_TABLE_SIZE];
} peer_list_t;

/// FUNCTIONS

void peer_list_init(peer_list_t *list);

void peer_list_destroy(peer_list_t *list);

int peer_list_check_present(peer_list_t *list, const char *username);

int peer_list_add_peer(peer_list_t *list, int new_peer_socket_fd, const char *username);

void peer_list_remove_peer(peer_list_t *list, const char *username);