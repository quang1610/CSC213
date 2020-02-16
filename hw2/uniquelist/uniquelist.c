#include "uniquelist.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Author Quang Nguyen
/// Initialize a new uniquelist
/*
 * In this function, we initialize s's values
 */
void uniquelist_init(uniquelist_t *s) {
    s->size = 0;
    s->next = NULL;
}

/// Destroy a uniquelist
/// recursively destroy node list of s
void uniquelist_node_destroy(uniquelist_node_t *node) {
    assert(node != NULL);
    if (node->next != NULL) {
        uniquelist_node_destroy(node->next);
    }
    free(node);
}

/*
 * In this function, we use recursive method in order to
 */
void uniquelist_destroy(uniquelist_t *s) {
    if (s == NULL || s->size == 0) {
        /// nothing to destroy
        return;
    }
    /// recursively destroy node list of s
    uniquelist_node_destroy(s->next);
    s->size = 0;
    s->next = NULL;
}

/// Add an element to a uniquelist, unless it's already in the uniquelist
/*
 * In this function, we have two big case:
 *  - In 1st case: there haven't been any numbers in the list yet. We add new uniquelist_node to the list.
 *
 *  - In 2nd case: we have to loop over the list and see if there are any duplicated numbers. If we reach the end of the
 *  list and the input number n is unique then we add it to the list.
 *
 */
void uniquelist_insert(uniquelist_t *s, int n) {
    assert(s != NULL);
    /// case 1
    if (s->size == 0) {
        uniquelist_node_t *new_node = malloc(sizeof(uniquelist_node_t));
        new_node->num = n;
        new_node->pos = 0;
        new_node->next = NULL;

        s->next = new_node;
        s->size = s->size + 1;
    } else {
        /// case 2
        uniquelist_node_t *cursor = s->next;
        do {
            if (cursor->num == n) {
                /// if there is a duplicated number
                return;
            } else if (cursor->next == NULL) {
                /// if there aren't any duplicated numbers and we are at the end of list
                uniquelist_node_t *new_node = malloc(sizeof(uniquelist_node_t));
                new_node->next = NULL;
                new_node->num = n;
                new_node->pos = cursor->pos + 1;

                cursor->next = new_node;

                s->size = s->size + 1;
                return;
            }
            cursor = cursor->next;
        } while (cursor != NULL);   /// keep looping till the end of the list
    }
}

/// Print all the numbers in a uniquelist
/*
 * In this function, we loop through the list from the beginning til end and print out each node's number
 */
void uniquelist_print_all(uniquelist_t *s) {
    uniquelist_node_t *cursor = s->next;
    while (cursor != NULL && cursor->pos != -1) {
        printf("%d ", cursor->num);
        cursor = cursor->next;

        if (cursor == NULL) printf("\n");
    }
}
