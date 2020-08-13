//
// Created by Nguyễn Đức Quang on 5/6/20.
//

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "peer_list.h"
#include "hashcode.h"

void peer_list_init(peer_list_t *list, const char *my_username) {
    strcpy(&list->my_username[0], my_username);

    for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
        pthread_mutex_init(&list->mutex[i], NULL);
        list->table[i] = NULL;
        list->size[i] = 0;
    }
}

void peer_list_destroy(peer_list_t *list) {
    for (int i = 0; i < PEER_LIST_TABLE_SIZE; i++) {
        /// acquire the lock and set table entry to NULL
        pthread_mutex_lock(&list->mutex[i]);

        peer_t *cursor = list->table[i];
        list->table[i] = NULL;
        list->size[i] = 0;

        pthread_mutex_unlock(&list->mutex[i]);

        /// actual clean up
        while (cursor != NULL) {
            peer_t *temp = cursor;
            cursor = cursor->next;

            close(temp->to_peer);
            close(temp->from_peer);
            close(temp->socket_fd);

            free(temp);
        }
    }
}

int peer_list_check_present(peer_list_t *list, const char *username) {
    unsigned long username_hashcode = hashcode(username);
    int index = (int) (username_hashcode % PEER_LIST_TABLE_SIZE);

    pthread_mutex_lock(&list->mutex[index]);

    peer_t *cursor = list->table[index];
    while (cursor != NULL) {
        if (cursor->hashcode == username_hashcode && strcmp(username, cursor->peer_username) == 0) {
            pthread_mutex_unlock(&list->mutex[index]);
            return IN_PEER_LIST;
        }
        cursor = cursor->next;
    }

    pthread_mutex_unlock(&list->mutex[index]);
    return NOT_IN_PEER_LIST;
}

int peer_list_find_peer_fd(peer_list_t *list, const char *username) {
    /// not gonna return myself
    if (strcmp(list->my_username, username) == 0)
        return 0;

    /// setting hashcode and index
    unsigned long username_hashcode = hashcode(username);
    int index = (int) (username_hashcode % PEER_LIST_TABLE_SIZE);

    pthread_mutex_lock(&list->mutex[index]);
    peer_t *cursor = list->table[index];

    while (cursor != 0) {
        if (username_hashcode == cursor->hashcode && strcmp(cursor->peer_username, username) == 0) {
            int return_fd = cursor->to_peer;
            pthread_mutex_unlock(&list->mutex[index]);
            return return_fd;
        }
        cursor = cursor->next;
    }

    pthread_mutex_unlock(&list->mutex[index]);
    return -1;
}

int peer_list_add_peer(peer_list_t *list, int new_peer_socket_fd, const char *username) {
    /// avoid sending message to oneself
    if (strcmp(username, list->my_username) == 0) {
        return PEER_LIST_DUPLICATE_ADD;
    }

    /// finding hashcode and index
    unsigned long username_hashcode = hashcode(username);
    int index = (int) (username_hashcode % PEER_LIST_TABLE_SIZE);

    pthread_mutex_lock(&list->mutex[index]);

    /// check if peer has already been added
    peer_t *cursor = list->table[index];
    while (cursor != NULL) {
        if (cursor->hashcode == username_hashcode && strcmp(username, cursor->peer_username) == 0) {
            pthread_mutex_unlock(&list->mutex[index]);
            return PEER_LIST_DUPLICATE_ADD;
        }
        cursor = cursor->next;
    }

    /// Now we are sure that this is a new peer, we need to add them
    /// set up to and from stream
    int to_peer = dup(new_peer_socket_fd);
    int from_peer = dup(new_peer_socket_fd);

    /// create new peer
    peer_t *new_peer = (peer_t *) malloc(sizeof(peer_t));
    if (new_peer == NULL) {
        pthread_mutex_unlock(&list->mutex[index]);
        return PEER_LIST_ADD_FAIL;
    }
    new_peer->socket_fd = new_peer_socket_fd;
    new_peer->to_peer = to_peer;
    new_peer->from_peer = from_peer;
    strcpy(&new_peer->peer_username[0], username);
    new_peer->hashcode = username_hashcode;
    new_peer->next = NULL;
    new_peer->previous = NULL;

    /// Adding to the table
    if (list->table[index] == NULL) {
        /// current table entry is empty so we just add new peer to it
        list->table[index] = new_peer;
        list->size[index] += 1;
    } else {
        /// adding
        new_peer->next = list->table[index];
        list->table[index]->previous = new_peer;
        list->table[index] = new_peer;

        list->size[index] += 1;
    }
    pthread_mutex_unlock(&list->mutex[index]);
    return PEER_LIST_ADD_SUCCESSFUL;
}

void peer_list_remove_peer(peer_list_t *list, const char *username) {
    /// avoid removing yourself
    if(strcmp(username, list->my_username) == 0) {
        return;
    }

    // finding hashcode and index
    unsigned long username_hashcode = hashcode(username);
    int index = (int) (username_hashcode % PEER_LIST_TABLE_SIZE);

    pthread_mutex_lock(&list->mutex[index]);

    peer_t *cursor = list->table[index];

    while (cursor != NULL) {
        if (cursor->hashcode == username_hashcode && strcmp(username, cursor->peer_username) == 0) {
            /// if we found the one we need to remove!
            /// check if it has previous node
            if (cursor->previous != NULL) {
                cursor->previous->next = cursor->next;
            } else {
                /// we remove the first item of the bucket
                list->table[index] = cursor->next;
            }
            /// check if it has next node
            if (cursor->next != NULL) {
                cursor->next->previous = cursor->previous;
            }

            /// after finishing wiring surrounding nodes, we remove the current cursor
            close(cursor->to_peer);
            close(cursor->from_peer);
            close(cursor->socket_fd);
            free(cursor);

            list->size[index] -= 1;

            pthread_mutex_unlock(&list->mutex[index]);
            return;
        }
        cursor = cursor->next;
    }

    pthread_mutex_unlock(&list->mutex[index]);
}