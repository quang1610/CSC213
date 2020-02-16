#define _GNU_SOURCE
#include <unistd.h>
#include <sys/mman.h>

#include "lazycopy.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

// *********************************************************************************************************************
/// book keeping list
typedef struct node  {
  void * chunk_addr;
  struct node * next_node;
  struct node * pre_node;
} chunk_addr_node;

chunk_addr_node* head = NULL;

void insert_address_node (void * new_address) {
  chunk_addr_node * new_node = malloc(sizeof(chunk_addr_node));
  new_node->chunk_addr = new_address;
  new_node->next_node = head;
  new_node->pre_node = NULL;

  if(head != NULL)
  head->pre_node = new_node;

  head = new_node;
}

/// Look for the chunk that contains address
void * search_chunk(void * address, int remove_chunk_addr) {
  chunk_addr_node * cursor = head;

  while (cursor != NULL) {
    intptr_t current_chunk_addr_i = (intptr_t)cursor->chunk_addr;
    intptr_t address_i = (intptr_t) address;
    if (current_chunk_addr_i <= address_i && address_i <= (current_chunk_addr_i + CHUNKSIZE)) {
      void * return_addr = cursor->chunk_addr;

      // remove the node from list
      if (remove_chunk_addr == REMOVE_CHUNK) {
        cursor->pre_node->next_node = cursor->next_node;
        if(cursor->next_node != NULL) {
          cursor->next_node->pre_node = cursor->pre_node;
        }

        free(cursor);
      }
 
      return return_addr;
    }
    cursor = cursor->next_node;
  }
  return NULL;
}

// *********************************************************************************************************************

void segfault_handler(int signal, siginfo_t* info, void* ctx) {
  // do something with the memory chunk being violated.
  void* seg_fault_addr = info->si_addr;

  // find the chunk that has seg-fault
  void* chunk_address = search_chunk(seg_fault_addr, REMOVE_CHUNK);

  if(chunk_address != NULL) {
    void* local_copy = malloc(CHUNKSIZE*sizeof(char));

    memcpy(local_copy, chunk_address, sizeof(char) * CHUNKSIZE);
    
    void * new_chunk_address = mmap(chunk_address, CHUNKSIZE, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED | MAP_FIXED, -1, 0);
    // Check for an error
    if(new_chunk_address == MAP_FAILED) {
      perror("mmap failed in chunk_alloc");
      exit(2);
    }

    memcpy(new_chunk_address, local_copy, sizeof(char) * CHUNKSIZE);
    
  } else {
    // segfault is not due to access read-only chunk
     printf("We occur a fatal segfault!\n");
     exit(1);
  }
}

/**
 * This function will be called at startup so you can set up a signal handler.
 */
__attribute__((constructor)) void chunk_startup() {
  chunk_addr_node * chunks_list = (chunk_addr_node*) malloc(sizeof(chunk_addr_node));
  srand(time(NULL));
  
  // Make a sigaction struct to hold our signal handler information
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sa.sa_sigaction = segfault_handler;
  sa.sa_flags = SA_SIGINFO;
  
  // Set the signal handler, checking for errors
  if(sigaction(SIGSEGV, &sa, NULL) != 0) {
    perror("sigaction failed");
    exit(2);
  } 
}

/**
 * This function should return a new chunk of memory for use.
 *
 * \returns a pointer to the beginning of a 64KB chunk of memory that can be read, written, and copied
 */
void* chunk_alloc() {
  // Call mmap to request a new chunk of memory. See comments below for description of arguments.
  void* result = mmap(NULL, CHUNKSIZE, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  // Arguments:
  //   NULL: this is the address we'd like to map at. By passing null, we're asking the OS to decide.
  //   CHUNKSIZE: This is the size of the new mapping in bytes.
  //   PROT_READ | PROT_WRITE: This makes the new reading readable and writable
  //   MAP_ANONYMOUS | MAP_SHARED: This mapes a new mapping to cleared memory instead of a file,
  //                               which is another use for mmap. MAP_SHARED makes it possible for us
  //                               to create shared mappings to the same memory.
  //   -1: We're not connecting this memory to a file, so we pass -1 here.
  //   0: This doesn't matter. It would be the offset into a file, but we aren't using one.
  
  // Check for an error
  if(result == MAP_FAILED) {
    perror("mmap failed in chunk_alloc");
    exit(2);
  }
  
  // Everything is okay. Return the pointer.
  return result;
}

/**
 * Create a copy of a chunk by copying values eagerly.
 *
 * \param chunk This parameter points to the beginning of a chunk returned from chunk_alloc()
 * \returns a pointer to the beginning of a new chunk that holds a copy of the values from
 *   the original chunk.
 */
void* chunk_copy_eager(void* chunk) {
  // First, we'll allocate a new chunk to copy to
  void* new_chunk = chunk_alloc();
  
  // Now copy the data
  memcpy(new_chunk, chunk, CHUNKSIZE);
  
  // Return the new chunk
  return new_chunk;
}

/**
 * Create a copy of a chunk by copying values lazily.
 *
 * \param chunk This parameter points to the beginning of a chunk returned from chunk_alloc()
 * \returns a pointer to the beginning of a new chunk that holds a copy of the values from
 *   the original chunk.
 */
void* chunk_copy_lazy(void* chunk) {
  // Just to make sure your code works, this implementation currently calls the eager copy version
  //return chunk_copy_eager(chunk);
  
  // Your implementation should do the following:
  // 1. Use mremap to create a duplicate mapping of the chunk passed in
  void* new_chunk = mremap(chunk, 0, CHUNKSIZE, MREMAP_MAYMOVE);
  // Check for an error
  if(new_chunk == MAP_FAILED) {
    perror("mmap failed in chunk_alloc");
    exit(2);
  }
  // 2. Mark both mappings as read-only
  mprotect(new_chunk, CHUNKSIZE, PROT_READ);
  mprotect(chunk, CHUNKSIZE, PROT_READ);
  
  // 3. Keep some record of both lazy copies so you can make them writable later.
  //    At a minimum, you'll need to know where the chunk begins and ends.
  insert_address_node(chunk);
  insert_address_node(new_chunk);
  
  
  // Later, if either copy is written to you will need to:
  // 1. Save the contents of the chunk elsewhere (a local array works well)
  // 2. Use mmap to make a writable mapping at the location of the chunk that was written
  // 3. Restore the contents of the chunk to the new writable mapping
  return new_chunk;
}
