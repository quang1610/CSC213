#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct node {
    char str[10];
} node_t;

int main() {
    node_t *new_node = malloc(sizeof(node_t));

    strcpy(&(new_node->str[0]), "hello\n");

    printf("%s", new_node->str);
}

