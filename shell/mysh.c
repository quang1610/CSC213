// Author: Quang Nguyen & Yolanda Jiang
// Date: Feb.6

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

# define BACKGROUND 0
# define SEQUENTIAL 1

// This is the maximum number of arguments your shell should handle for one command
#define MAX_ARGS 128
#define DELIM "\n \t"

void execute_cmd (char * line, int exe_mode) {
  char** tokens = (char**) calloc( MAX_ARGS, sizeof(char*));
  int num_args = 0; // the number of parsed tokens from line
    
  // split the commmand:
  char *saveptr = line;
  for(int i = 0; i < MAX_ARGS; i++) {
    tokens[i] = strtok_r(saveptr, DELIM, &saveptr);
    num_args++;
    // ending with NULL entry
    if (tokens[i] == NULL){
      tokens[i] = NULL;
      break;
    }
  }
  
  // execute the command line
  if (tokens[0] == NULL)
    return;
  else if (strcmp(tokens[0], "cd") == 0)
    if(tokens[1] == NULL || tokens[1][0] == '~') {
      chdir(getenv("HOME"));
    } else {
      chdir(tokens[1]);
    }
  else if (strcmp(tokens[0], "exit") == 0)
    exit(1);
  else {
    int rc = fork();
    if (rc < 0) {
      perror("Cannot evoke execution!");
    } else if (rc == 0) {
      // running child process
      execvp(tokens[0], tokens);
    } else {
      int status;
      int wait_rc = waitpid(-1, &status, WNOHANG);
      
      // running parent process, parent will wait till child finish
      if (exe_mode == SEQUENTIAL) {
        while (wait_rc >= 0) {
          if (wait_rc > 0)
            printf("Child process %d exited with status %d\n", wait_rc, status);
          wait_rc = waitpid(-1, &status, WNOHANG);
        }
      } else {
        if(wait_rc > 0)
          printf("Child process %d exited with status %d\n", wait_rc, status);
      } 
    }
  }
}


int main(int argc, char** argv) {
  // If there was a command line option passed in, use that file instead of stdin
  if(argc == 2) {
    // Try to open the file
    int new_input = open(argv[1], O_RDONLY);
    if(new_input == -1) {
      fprintf(stderr, "Failed to open input file %s\n", argv[1]);
      exit(1);
    }
    
    // Now swap this file in and use it as stdin
    if(dup2(new_input, STDIN_FILENO) == -1) {
      fprintf(stderr, "Failed to set new file as input\n");
      exit(2);
    }
  }
  
  char* line = NULL;    // Pointer that will hold the line we read in
  size_t line_size = 0; // The number of bytes available in line

  // Loop forever
  while(true) {
    // Print the shell prompt
    printf("$ ");
    
    // Get a line of stdin, storing the string pointer in line
    if(getline(&line, &line_size, stdin) == -1) {
      if(errno == EINVAL) {
        perror("Unable to read command line");
        exit(2);
      } else {
        // Must have been end of file (ctrl+D)
        printf("\nShutting down...\n");
        
        // Exit the infinite loop
        break;
      }
    }

    char * current_position = line;
    while(true) {
       if (*current_position == '\n') break;
      char* delim_position = strpbrk(current_position, ";&");
      if(delim_position == NULL) {
        // There were no more delimeters or we are in the end of the cmd
        execute_cmd(current_position, SEQUENTIAL);
	break;
      } else {
        // There was a delimeter. First, save it.
        char delim = *delim_position;

        // Overwrite the delimeter with a null terminator so we can print just this fragment
        if (delim == ';') {
          *delim_position = '\0';
          execute_cmd(current_position, SEQUENTIAL);
        } else if (delim == '&') {
          *delim_position = '\0';
          execute_cmd(current_position, BACKGROUND);
        }
      }
      // Move our current position in the string to one character past the delimeter
      current_position = delim_position + 1;
    }

    // wait for the rest child processes to terminate 
    int status;
    int wait_rc = waitpid(0, &status, WNOHANG);
    while (wait_rc >= 0) {
      if(wait_rc > 0)
        printf("Background child process %d finally exited with status %d\n", wait_rc, status);
      wait_rc = waitpid(0, &status, WNOHANG);
    }
  }
  
  // If we read in at least one line, free this space
  if(line != NULL) {
    free(line);
  }
  
  return 0;
}
