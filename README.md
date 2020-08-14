# CSC213 Project
All projects developed by me during **CSC213-Operating System**. Most projects used C/C++ standard libraries and APIs.

## Subdirectories
### 1. Archive Printer
- Program in this project reads the contents of the a UNIX archive (.ar file).

### 2. Parallel LetterCount
- Program in this project uses parallel programming to count the occurrence of every letter in an input file. 

### 3. Memory Allocator
- This project implements a "Big Bag of Pages" memory allocator in C. The program requests large blocks of memory. Each 
block is divided into fixed-size chunk of memory to serve back to the requester. In short, we make malloc() and free()
from scratch.

### 4. Ngram Generator
- This program generates all possible ngrams from the text input. 

### 5. Password Cracker
- This is a multi-thread program that brute forces 6-character passwords by hashing guessed passwords and comparing them 
to the hash of target passwords.

### 6. shell
- This program emulates an actual linux terminal. We practice managing process using fork() and and exec() system calls.

### 7. Virtual Memory 
- This program implements a lazy copy technique commonly used in memory virtualization using memory API (mmap, mprotect, mremap...),
and POSIX's signal feature.

### 8. worm
- This program implements a Round-Robin process scheduler, which has to deal with graphic update, user input...) and later, applies 
this to the classic worm game.

### 9. PeerToPeer Chat
- This program use web socket API and multi thread programming to make a chat P2P program. In this chat program, every node acts
both as a server and a client. There is no central server. 



 