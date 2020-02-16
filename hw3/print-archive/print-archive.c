#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FILE_HEADER_SIZE 60

typedef struct __attribute__((packed)) file_header {
    uint8_t file_id[16];
    uint8_t file_time_stamp[12];
    uint8_t file_owner_id[6];
    uint8_t file_group_id[6];
    uint8_t file_mode[8];
    uint8_t file_size[10];
    uint8_t file_end_char[2];
}File_header;
void print_contents(uint8_t* data, size_t size);

int main(int argc, char** argv) {
  // Make sure we have a file input
  if(argc != 2) {
    fprintf(stderr, "Please specify an input filename.\n");
    exit(1);
  }
  
  // Try to open the file
  FILE* input = fopen(argv[1], "r");
  if(input == NULL) {
    perror("Unable to open input file");
    exit(1);
  }
  
  // Seek to the end of the file so we can get its size
  if(fseek(input, 0, SEEK_END) != 0) {
    perror("Unable to seek to end of file");
    exit(2);
  }
  
  // Get the size of the file
  size_t size = ftell(input);
  
  // Seek back to the beginning of the file
  if(fseek(input, 0, SEEK_SET) != 0) {
    perror("Unable to seek to beginning of file");
    exit(2);
  }
  
  // Allocate a buffer to hold the file contents. We know the size in bytes, so
  // there's no need to multiply to get the size we pass to malloc in this case.
  uint8_t* data = malloc(size);
  
  // Read the file contents
  if(fread(data, 1, size, input) != size) {
    fprintf(stderr, "Failed to read entire file\n");
    exit(2);
  }
  
  // Make sure the file starts with the .ar file signature
  if(memcmp(data, "!<arch>\n", 8) != 0) {
    fprintf(stderr, "Input file is not in valid .ar format\n");
    exit(1);
  }
  
  // Call the code to print the archive contents
  print_contents(data, size);
  
  // Clean up
  free(data);
  fclose(input);
  
  return 0;
}

/**
 * This function should print the name of each file in the archive followed by its contents.
 *
 * \param data This is a pointer to the first byte in the file.
 * \param size This is the number of bytes in the file.
 */
void print_contents(uint8_t* data, size_t size) {
    printf("original size %lu\n", size);
    uint8_t * cursor = data + 8;
    size_t read_size = 8;
    File_header local_file_header;
    // size_t *file_size = malloc(sizeof(size_t)); // this is the size to one file in archive. Will be assigned later.

    while (read_size < size ) {
        memcpy(&local_file_header, cursor, FILE_HEADER_SIZE);

        /// formating file name:
        // replace the '/' character with '\0'
        char * end_file_id_pos = strchr((char*)local_file_header.file_id, '/');
        *end_file_id_pos = '\0';

        // printout the file name
        printf("%s\n", (char*)local_file_header.file_id);

        /// reading the file content:
        // first, we read the file size into file_size_str and then convert file_size_str to numerical file_size
        char file_size_str[11];
        file_size_str[10] = '\0';
        memcpy(&(file_size_str[0]), local_file_header.file_size, 10);
        size_t file_size = strtol(file_size_str, NULL, 10);

        // next, we determine the start of file content:
        uint8_t *content_pos = cursor + FILE_HEADER_SIZE;
        fwrite((char*) content_pos, sizeof(char), file_size, stdout);

        /// update remain_size and cursor
        if (file_size % 2 == 0) {
            read_size = read_size + FILE_HEADER_SIZE + file_size;
            if(read_size < size) {
                cursor = cursor + FILE_HEADER_SIZE + file_size;
            }
        } else {
            read_size = read_size + FILE_HEADER_SIZE + file_size + 1;
            if(read_size < size) {
                cursor = cursor + FILE_HEADER_SIZE + file_size + 1;
            }
        }
        printf("\n");
    }
}
