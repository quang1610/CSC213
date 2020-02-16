#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "uniquelist.h"

int main(int argc, char** argv) {
  // Make a uniquelist and initialize it
  uniquelist_t nums;
  uniquelist_init(&nums);
  
  // Keep reading input until we hit an invalid value or the end of the input
  int n;
  int rc;
  while((rc = scanf("%d", &n)) == 1) {
    uniquelist_insert(&nums, n);
  }
  
  // Print the uniquelist (no duplicates!)
  uniquelist_print_all(&nums);
  
  // Clean up
  uniquelist_destroy(&nums);
  
  return 0;
}


