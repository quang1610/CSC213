#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "socket.h"
#include "ui.h"
#include "utility.h"
#include "get_ip.h"
#include "p2p_thread.h"

/// SUPPORT FUNCTIONS
// Keep the username in a global so we can access it from the callback
const char *username;

char my_username[USERNAME_LEN];
char my_server_name[SERVER_NAME_LEN];
unsigned short my_port;

int terminate = TRUE;
peer_list_t my_peer_list;
mess_record_t my_mess_record;

// This function is run whenever the user hits enter after typing a message
void input_callback(const char *message) {
    if (terminate == FALSE) {
        if (strcmp(message, ":quit") == 0 || strcmp(message, ":q") == 0) {
            /// sending removal request to peer
            message_t *new_message = message_generate(TYPE_REMOVE_PEER, my_username, NULL, my_server_name, my_port);
            send_all(new_message, &my_peer_list);

            /// terminate the thread
            terminate = TRUE;

            /// free the resource
            peer_list_destroy(&my_peer_list);
            mess_record_destroy(&my_mess_record);

            /// exit
            ui_exit();
        } else {
            message_t *new_message = message_generate(TYPE_NORMAL, my_username, message, my_server_name, my_port);
            send_all(new_message, &my_peer_list);
            free(new_message);

            ui_display(username, message);
        }
    }
}

/// MAIN FUNTIONS
int main(int argc, char **argv) {
    // Make sure the arguments include a username
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "Usage: %s <username> [<peer> <port number>]\n", argv[0]);
        exit(1);
    }

    // Save the username in a global
    /// set up username
    username = argv[1];
    strcpy(my_username, argv[1]);

    /// set up servername
    char *IP = get_ip();
    strcpy(my_server_name, IP);

    /// set up some peer list and message record
    peer_list_init(&my_peer_list, my_username);
    mess_record_init(&my_mess_record, my_username);

    /// this will be the host of the room
    /// create new server
    my_port = 0;
    int my_server_socket_fd = server_socket_open(&my_port);
    if (my_server_socket_fd == -1) {
        perror("fail to create server!\n");
        exit(2);
    }

    /// start listening for new connections
    if (listen(my_server_socket_fd, MAX_QUEUE_CONNECTION)) {
        perror("listen failed\n");
        exit(2);
    }

    /// create accepting thread to accept new clients
    terminate = FALSE;

    reception_args_t args;

    args.terminate = &terminate;
    args.server_socket_fd = my_server_socket_fd;
    args.mess_record = &my_mess_record;
    args.peer_list = &my_peer_list;

    pthread_t accepting_thread;
    pthread_create(&accepting_thread, NULL, receiving_new_connection_worker, &args);


    if (argc == 4) {
        /// connect to someone
        // Unpack arguments
        char *peer_hostname = argv[2];
        unsigned short peer_port = atoi(argv[3]);

        /// making the connection
        int peer_socket = socket_connect(peer_hostname, peer_port);
        if (peer_socket == -1) {
            perror("making friend fail!\n");
            exit(2);
        }

        /// sending out peer request
        char mess_buff[MESSAGE_LEN];
        sprintf(mess_buff, "%s at address %s is listening at %d\n", my_username, my_server_name, my_port);
        message_t *new_peer_request = message_generate(TYPE_ADD_PEER, my_username, mess_buff, my_server_name, my_port);

        send_message(new_peer_request, peer_socket);
    }

    // Set up the user interface. The input_callback function will be called
    // each time the user hits enter to send a message.
    ui_init(input_callback);

    // Once the UI is running, you can use it to display log messages
    char mess_buff[MESSAGE_LEN];
    sprintf(mess_buff, "%s at address %s is listening at %d\n", my_username, my_server_name, my_port);
    ui_display("INFO", mess_buff);

    // Run the UI loop. This function only returns once we call ui_stop() somewhere in the program.
    ui_run();

    return 0;
}
