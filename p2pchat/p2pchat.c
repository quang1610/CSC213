#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "socket.h"
#include "ui.h"

// Keep the username in a global so we can access it from the callback
const char* username;

// This function is run whenever the user hits enter after typing a message
void input_callback(const char* message) {
  if(strcmp(message, ":quit") == 0 || strcmp(message, ":q") == 0) {
    ui_exit();
  } else {
    ui_display(username, message);
  }
}

int main(int argc, char** argv) {
  // Make sure the arguments include a username
  if(argc != 2 && argc != 4) {
    fprintf(stderr, "Usage: %s <username> [<peer> <port number>]\n", argv[0]);
    exit(1);
  }
  
  // Save the username in a global
  username = argv[1];
  
  // TODO: Set up a server socket to accept incoming connections
  
  // Did the user specify a peer we should connect to?
  if(argc == 4) {
    // Unpack arguments
    char* peer_hostname = argv[2];
    unsigned short peer_port = atoi(argv[3]);
    
    // TODO: Connect to another peer in the chat network
  }
  
  // Set up the user interface. The input_callback function will be called
  // each time the user hits enter to send a message.
  ui_init(input_callback);
  
  // Once the UI is running, you can use it to display log messages
  ui_display("INFO", "This is a handy log message.");
  
  // Run the UI loop. This function only returns once we call ui_stop() somewhere in the program.
  ui_run();
  
  return 0;
}
