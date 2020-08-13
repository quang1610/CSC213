#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <time.h>

const char* segfault_message[] = { "Seg fault: You can do it!\n",
                               "Seg fault: Keep trying!\n",
                               "Seg fault: You're amazing!\n",
                               "Seg fault: You're a great programmer!\n",
                               "Seg fault: You'll figure it out!\n",
                               "Seg fault: I love you!\n",
                               "Seg fault: Keet at it!\n",
                               "Seg fault: It happens to the best of us!\n" };

void segfault_handler(int signal, siginfo_t* info, void* ctx) {
  int message_id =  rand() % 8;
  printf("%s", segfault_message[message_id]);

  // do something with the memory chunk being violated.
  
  exit(1);
}

__attribute__((constructor)) void init() {
  printf("This code runs at program startup\n");
  srand(time(NULL));
  
  // Make a sigaction struct to hold our signal handler information
  struct sigaction sa;
  memset(&sa, 0, sizeof(struct sigaction));
  sa.sa_sigaction = segfault_handler;
  sa.sa_flags = SA_SIGINFO;
  
  // Set the signal handler, checking for errors
  if(sigaction(SIGSEGV, &sa, NULL) != 0) {
    perror("sigaction failed");
    exit(2);
  } 
}

