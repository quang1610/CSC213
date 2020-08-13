CC := clang
CFLAGS := -g -Wall -Werror -Wno-unused-function -Wno-unused-variable

all: p2pchat

clean:
	rm -rf p2pchat p2pchat.dSYM

p2pchat: p2pchat.c ui.c ui.h message.c message.h get_ip.c get_ip.h p2p_thread.c p2p_thread.h hashcode.c hashcode.h peer_list.c peer_list.h utility.h
	$(CC) $(CFLAGS) -o p2pchat p2pchat.c ui.c message.c p2p_thread.c get_ip.c hashcode.c peer_list.c -lform -lncurses -lpthread
