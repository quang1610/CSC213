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

/**
 * Calculate the object size based on the input integer. The return value should be a multiple of 16 and
 * exponent of 2 and smallest possible such that it is >= input integer
 */
int round_up_multiple_16(int size) {
  int return_value = 16;
  while (size > return_value) {
    return_value *= 2;
  }
  return return_value;
}

/**
 * find_free_list_ind_base_16 calculate which type of object does input size fall into. if size <= 16 then 
 * return 0. If 16 < size <= 32, return 1. If 32 < size <= 64, return 2...
 */
int find_free_list_ind_base16 (int size) {
  int return_value = 0;
  while (size > 16) {
    size = size / 2;
    return_value ++;
  }
  return return_value;
}

/**
 * Allocate space on the heap.
 * \param size  The minimium number of bytes that must be allocated
 * \returns     A pointer to the beginning of the allocated space.
 *              This function may return NULL when an error occurs.
 */
void* xxmalloc(size_t size) {
  // Round the size up to the next multiple of the page size
  size = ROUND_UP(size, PAGE_SIZE);
  
  // Request memory from the operating system in page-sized chunks
  void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

  // Check for errors
  if(p == MAP_FAILED) {
    fputs("mmap failed! Giving up.\n", stderr);
    exit(2);
  }
  
  return p;
}

/**
 * Free space occupied by a heap object.
 * \param ptr   A pointer somewhere inside the object that is being freed
 */
void xxfree(void* ptr) {
  // Don't free NULL!
  if(ptr == NULL) return;
  
  // TODO: Complete this function
}

/**
 * Get the available size of an allocated object
 * \param ptr   A pointer somewhere inside the allocated object
 * \returns     The number of bytes available for use in this object
 */
size_t xxmalloc_usable_size(void* ptr) {
  // TODO: Complete this function
  return 0;
}

