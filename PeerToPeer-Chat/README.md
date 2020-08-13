# Peer-to-Peer Chat program
In this P2P chat program, there is no server. Each node inside acts as both the server and the client. It is implemented 
using C Socket API, hash algorithm, and multi thread programming. 

# Instruction
## 1. Compilation:
> rm p2pchat
> make 

## 2. Starting the chat:
> ./p2pchat <Username>

For example: 
*> ./p2pchat Alice*
The terminal will print out message such as INFO: Alice at address 127.0.0.1 is listening at 55107. 
55107 is the port that Alice is currently listening to. Other node can connect to the network via this port.

## 3. Joining the chat:
> ./p2pchat <Username> <hosting IP> <port that any node is listening to>

For example: *./p2pchat Bob localhost 55107

## 4. Quit the chat:
To quit the chat, type: *:p* into the chat box.

# Demo
![](Quang-P2PChat-Demo.GIF)


