#define _GNU_SOURCE

#include <assert.h>
#include <malloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

// The minimum size returned by malloc
#define MIN_MALLOC_SIZE 16

// Round a value x up to the next multiple of y
#define ROUND_UP(x,y) ((x) % (y) == 0 ? (x) : (x) + ((y) - (x) % (y)))

// The size of a single page of memory, in bytes
#define PAGE_SIZE 0x1000

// function prototypes
int round_up_multiple_16(int size);
int find_free_lists_index (size_t size);
void* xxmalloc(size_t size);
void xxfree(void* ptr);
size_t xxmalloc_usable_size(void* ptr);

// support data structures
typedef struct header {
  size_t size;
  long magic_number;
} header_t;

typedef struct node {
  struct node* next;
} node_t;

node_t* free_linked_lists[8];

//****************************************************************************************************************
/**
 * Initialized chunks and blocks of memory inside each chunk.
 */
__attribute__((constructor)) void init_heap() {
  size_t current_size = MIN_MALLOC_SIZE / 2;
  
  for (int i = 0; i < 8; i ++) {
    current_size *= 2;
    void * new_chunk =  mmap(NULL, PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);

    /// initialize the header
    header_t* head = (header_t*) new_chunk;
    head->size = current_size;
    head->magic_number = (long) current_size;

    /// allocate the free linked list
    /// set the array elem point to first elem of free linked list
    free_linked_lists[i] = (node_t*) ((uintptr_t)new_chunk + current_size);
    node_t* cursor = free_linked_lists[i];
    long n_loops = (long) (PAGE_SIZE/current_size - 1);
    printf ("Debug, Allocating chunk for %lu, looping %lu time\n", current_size, n_loops);
    
    for (int i = 0; i < n_loops - 1; i ++) {
      cursor->next = (node_t*) ((uintptr_t)cursor + current_size);
      cursor = cursor->next;
    }
    cursor->next = NULL;
  }
}

//****************************************************************************************************************
/**
 * Calculate the object size based on the input integer. The return value should be a multiple of 16 and
 * exponent of 2 and smallest possible such that it is >= input integer
 */
int round_up_multiple_16(int size) {
  int return_value = MIN_MALLOC_SIZE;
  while (size > return_value) {
    return_value *= 2;
  }
  return return_value;
}

/**
 * find_free_list_type calculate which type of object does input size fall into. if size <= 16 then 
 * return 0. If 16 < size <= 32, return 1. If 32 < size <= 64, return 2...
 */
int find_free_lists_index (size_t size) {
  int return_value = 0;
  while (size > MIN_MALLOC_SIZE) {
    size = size / 2;
    return_value ++;
  }
  return return_value;
}

//****************************************************************************************************************
/**
 * Allocate space on the heap.
 * \param size  The minimium number of bytes that must be allocated
 * \returns     A pointer to the beginning of the allocated space.
 *              This function may return NULL when an error occurs.
 */
void* xxmalloc(size_t size) {
  if (size > PAGE_SIZE/2) {
    /// allocate object > 2048
    // Round the size up to the next multiple of the page size
    size = ROUND_UP(size, PAGE_SIZE);
    return mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
  } else {
    /// allocate object <= 2048
    int free_linked_lists_index = find_free_lists_index(size);
    node_t* p = free_linked_lists[free_linked_lists_index];

    // remove the list from the current linked_list
    if (p != NULL) free_linked_lists[free_linked_lists_index] = p -> next;
  
    return p;
  }
}

/**
 * Free space occupied by a heap object.
 * \param ptr   A pointer somewhere inside the object that is being freed
 */
void xxfree(void* ptr) {
  // Don't free NULL!
  if(ptr == NULL) return;

  /// find the chunk
  size_t object_size = xxmalloc_usable_size(ptr);

  if (object_size != 0) {
    int free_linked_lists_index = find_free_lists_index(object_size);

    node_t* freed_node = (node_t*) ptr;
    freed_node->next = free_linked_lists[free_linked_lists_index]->next;
    free_linked_lists[free_linked_lists_index] = freed_node;
  }
}

/**
 * Get the available size of an allocated object
 * \param ptr   A pointer somewhere inside the allocated object
 * \returns     The number of bytes available for use in this object
 */
size_t xxmalloc_usable_size(void* ptr) {
  header_t* head = (header_t*) ((uintptr_t)ptr - ((uintptr_t)ptr % PAGE_SIZE));
  size_t object_size = head->size;

  if(head->magic_number == (long)object_size) {
    return object_size;
  } else {
    return 0;
  }
}

