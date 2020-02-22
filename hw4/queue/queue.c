#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QUEUE_END -1

typedef struct queue_node {
    int value;
    struct queue_node *next;
} queue_node_t;

// This struct is where you should define your queue datatype. You may need to
// add another struct (e.g. a node struct) depending on how you choose to
// implement your queue.
typedef struct queue {
    int current_size;
    queue_node_t *start;
    queue_node_t *end;
} queue_t;

/**
 * Initialize a queue pointed to by the parameter q.
 * \param q  This points to allocated memory that should be initialized to an
 *           empty queue.
 */
void queue_init(queue_t *q) {
    q->current_size = 0;
    q->start = NULL;
    q->end = NULL;
}

/**
 * Add a value to a queue.
 * \param q       Points to a queue that has been initialized by queue_init.
 * \param value   The integer value to add to the queue
 */
void queue_put(queue_t *q, int value) {
    queue_node_t *new_node = malloc(sizeof(queue_node_t));

    if (new_node == NULL) return;

    new_node->value = value;
    new_node->next = NULL;

    if (q->current_size == 0) {
        q->start = new_node;
        q->end = new_node;
        q->current_size = q->current_size + 1;
    } else {
        q->end->next = new_node;
        q->end = new_node;
        q->current_size = q->current_size + 1;
    }
}

/**
 * Check if a queue is empty.
 * \param q   Points to a queue initialized by queue_init.
 * \returns   True if the queue is empty, otherwise false.
 */
bool queue_empty(queue_t *q) {
    return q->current_size == 0;
}

/**
 * Take a value from a queue.
 * \param q   Points to a queue initialized by queue_init.
 * \returns   The value that has been in the queue the longest time. If the
 *            queue is empty, return QUEUE_END.
 */
int queue_take(queue_t *q) {
    if(queue_empty(q)) {
        return QUEUE_END;
    } else {
        int return_value = q->start->value;

        queue_node_t* to_be_removed_node = q->start;
        q->start = q->start->next;

        free(to_be_removed_node);

        q->current_size = q->current_size - 1;
        return return_value;
    }
}

void node_destroy(queue_node_t *node) {
    queue_node_t *cursor = node;
    queue_node_t *temp = NULL;
    while(cursor != NULL) {
        temp = cursor;
        cursor = cursor->next;

        free(temp);
    }
}

/**
 * Free any memory allocated inside the queue data structure.
 * \param q   Points to a queue initialized by queue_init. The memory referenced
 *            by q should *NOT* be freed.
 */
void queue_destroy(queue_t *q) {
    node_destroy(q->start);

    q->start = NULL;
    q->end = NULL;
    q->current_size = 0;
}

int main(int argc, char **argv) {
    // Set up and initialize a queue
    queue_t q;
    queue_init(&q);

    // Read lines until the end of stdin
    char *line = NULL;
    size_t line_size = 0;
    while (getline(&line, &line_size, stdin) != EOF) {
        int num;

        // If the line has a take command, take a value from the queue
        if (strcmp(line, "take\n") == 0) {
            if (queue_empty(&q)) {
                printf("The queue is empty.\n");
            } else {
                printf("%d\n", queue_take(&q));
            }
        } else if (sscanf(line, "put %d\n", &num) == 1) {
            queue_put(&q, num);
        } else {
            printf("unrecognized command.\n");
        }
    }

    // Free the space allocated by getline
    free(line);

    // Clean up the queue
    queue_destroy(&q);

    return 0;
}
