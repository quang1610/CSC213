#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "socket.h"

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <server name> <port>\n", argv[0]);
        exit(1);
    }

    /// Read command line arguments
    char *server_name = argv[1];
    unsigned short port = atoi(argv[2]);

    /// Connect to the server
    int socket_fd = socket_connect(server_name, port);
    if (socket_fd == -1) {
        perror("Failed to connect");
        exit(2);
    }

    /// Set up file streams
    FILE *to_server = fdopen(dup(socket_fd), "wb");
    if (to_server == NULL) {
        perror("Failed to open stream to server");
        exit(2);
    }

    FILE *from_server = fdopen(dup(socket_fd), "rb");
    if (from_server == NULL) {
        perror("Failed to open stream from server");
        exit(2);
    }

    /// start sending and listening feedback from the server
    char client_message[BUFFER_LEN];
    char buffer[BUFFER_LEN];

    fflush(from_server);
    while (1) {
        // read the input from user and try to send it to the server
        if (fgets(client_message, BUFFER_LEN, stdin) != NULL) {
            fprintf(to_server, "%s", &(client_message[0]));
            fflush(to_server);
        }

        if (strcmp(buffer, QUIT) == 0) {
            break;
        } else {
            printf("Server sent: %s", buffer);
        }

        // Read a message from the server
        if (fgets(buffer, BUFFER_LEN, from_server) == NULL) {
            perror("Reading from server failed");
            exit(2);
        }
    }

    /// This is where we close the client side
    // Close file streams
    fclose(to_server);
    fclose(from_server);

    // Close socket
    close(socket_fd);

    return 0;
}


















