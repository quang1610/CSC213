//
// Created by Nguyễn Đức Quang on 5/7/20.
//
#include <unistd.h>
#include <assert.h>
#include <netdb.h>
#include <arpa/inet.h>

#include "get_ip.h"

char *get_ip() {
    char host[256];
    char *IP;
    struct hostent *host_entry;
    int hostname;

    // get host
    hostname = gethostname(host, sizeof(host));
    assert(hostname != -1);
    // get host entry
    host_entry = gethostbyname(host);
    assert(host_entry != NULL);
    //get IP
    IP = inet_ntoa(*((struct in_addr*) host_entry->h_addr_list[0]));

    return IP;
}
