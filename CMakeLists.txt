cmake_minimum_required(VERSION 3.15)
project(CSC213)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/local/include)
add_executable(CSC213 main.c p2pchat/peer_list.c p2pchat/hashcode.h p2pchat/hashcode.c p2pchat/message.h p2pchat/message.c p2pchat/p2p_thread.h p2pchat/p2p_thread.c p2pchat/get_ip.c p2pchat/get_ip.h)