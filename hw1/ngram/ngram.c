#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * @program name: n-gram
 * @author: Quang D Nguyen
 * @date: Jan 29 2020
 */

int main(int argc, char** argv) {
  // Make sure the program is run with an N parameter
  if(argc != 2) {
    fprintf(stderr, "Usage: %s N (N must be >= 1)\n", argv[0]);
    exit(1);
  }
  
  // Convert the N parameter to an integer
  int N = atoi(argv[1]);
  
  // Make sure N is >= 1
  if(N < 1) {
    fprintf(stderr, "Invalid N value %d\n", N);
    exit(1);
  }
  
  // Processing the input and printing out n-gram
  /*
   * Algorithm explanation: first we define read_size and buffer_size. buffer_size is used to allocate strings holder.
   * read_size is used to allocate buffer. holder has larger memory to hold the left-over of previous read if there are any
   * We will only read from std to buffer.
   *
   * First string holder is initialized to have length 0.
   *
   * First read:
   * A pointer cursor is used to point to beginning of holder. cursor will be used to traverse down the string to
   * print out n-gram.
   *
   * We use while loop and fgets() in order to read all the input from stdin to buffer. We check if all the input has length
   * < N by using (holder[0] == '\0' && strlen(buffer) < N) since holder[0] == '\0' only at the beginning of the program.
   * This boolean is away to indicate that this is the first read from stdin.
   *
   * Then we strcat(holder, buffer), then run a while loop to move the cursor down the string until we cannot print n-gram
   * any more. We will move the left-over and the null terminate to the beginning of holder. Holder now is a string ==
   * left-over string.
   *
   * From second read and on:
   * Rewind cursor
   *
   * This times holder[0] should not be '\0'. We read from stdin again to buffer. strcat(holder, buffer) the left-over
   * of previous read is now the beginning of the new read.
   *
   * Then repeat printing out n-gram like first loop.
   */

  size_t read_size = N * 2;
  size_t buffer_size = read_size + N;   // allocate more memory to hold left over from previous read
  char * holder = malloc(buffer_size * sizeof(char));
  char * buffer = malloc(read_size * sizeof(char));
  char * cursor = holder;

  holder[0] = '\0';     // initialized length of holder = 0

  while (fgets(buffer, read_size, stdin) != NULL) {
      // rewind cursor
      cursor = holder;

      // check if the input string has length < N, if so return
      if (holder[0] == '\0' && strlen(buffer) < N) {
          return 0;
      }

      // concat holder and buffer to merge left-over of previous read to current read.
      strcat(holder, buffer);

      // read n-gram until reach the end
      while (cursor[N - 1] != '\0') {
          fwrite(cursor, sizeof(char) * N, 1, stdout);
          printf("\n");

          cursor += 1;
      }

      // move the left-over and null character to the beginning of holder
      memmove(holder, cursor, sizeof(char) * (N));
  }

  return 0;
}
