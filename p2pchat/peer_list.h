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
/**
 * This struct holds information about a peer.
 */
typedef struct peer {
    unsigned long hashcode; // created from username

    int socket_fd;

    int to_peer;
    int from_peer;

    char peer_username[USERNAME_LEN];

    struct peer *next;
    struct peer *previous;
} peer_t;

/**
 * This is a concurrent hashtable that hold the information about peers
 */
typedef struct peer_list {
    char my_username[USERNAME_LEN];
    pthread_mutex_t mutex[PEER_LIST_TABLE_SIZE];
    peer_t *table[PEER_LIST_TABLE_SIZE];
    int size[PEER_LIST_TABLE_SIZE];
} peer_list_t;

/// FUNCTIONS
/**
 * Init the peer list. This function would init mutex locks, set table entries to NULL, set size entries to 0, and
 * set list->my_username to my_username.
 * @param list
 * @param my_username
 */
void peer_list_init(peer_list_t *list, const char *my_username);

/**
 * Destroy the list
 * @param list
 */
void peer_list_destroy(peer_list_t *list);

/**
 * Check if a peer with input username is in the list.
 * @param list
 * @param username
 * @return
 */
int peer_list_check_present(peer_list_t *list, const char *username);

/**
 * Add new peer into the list. This function would not add if the peer is current user or the peer was added.
 * @param list
 * @param new_peer_socket_fd
 * @param username
 * @return
 */
int peer_list_add_peer(peer_list_t *list, int new_peer_socket_fd, const char *username);

/**
 * remove a peer from the list.
 * @param list
 * @param username
 */
void peer_list_remove_peer(peer_list_t *list, const char *username);