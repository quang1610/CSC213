#ifndef UNIQUELIST_H
#define UNIQUELIST_H

// Author Quang Nguyen
/// actual node of the list
typedef struct uniquelist_node {
    int num;
    int pos;
    struct uniquelist_node * next;
} uniquelist_node_t;

/// this is the root of the node, contain some meta data about the uniquelist
typedef struct uniquelist {
    int size;
    uniquelist_node_t * next;
} uniquelist_t;

/// Initialize a new uniquelist
void uniquelist_init(uniquelist_t* s);

/// Destroy a uniquelist
void uniquelist_destroy(uniquelist_t* s);

/// Add an element to a uniquelist, unless it's already in the uniquelist
void uniquelist_insert(uniquelist_t* s, int n);

/// Print all the numbers in a uniquelist
void uniquelist_print_all(uniquelist_t* s);

#endif
