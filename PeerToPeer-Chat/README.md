# Peer-to-Peer Chat program
In this P2P chat program, there is no server Each node inside acts as both the server and the client. It is implemented using C Socket API, hash algorithm, thread-safe hash table and concurrent programming.

# Demo
![Peer to peer chat demo](Quang-P2PChat-Demo.GIF)

# How it works?
There are 3 main types of messages:
		+ Type 1: Normal text message
		+ Type 2: Add Peer message. if A sends B this message, B would connect to A where A is the server role, and B is client role.
		+ Type 3: Remove Peer message. This message is sent before a person completely leaves the room.

- Username must be unique although there hasn't a solution to check username yet.
- Each message has a hashcode and time stamp as a unique id. A person will record these hashcode and time stamp so that if he meets the message again, he would not forward it. This would avoid endless sending message loop in the network. Only new messages are forward.
- Each person has a list of their peers to whom they will forward new messages. Each person on this list has a unique username!

- Protocol: Here is how the network works:
    + First, person A initializes the chat room by simply run the program with his username as the argument. A would constantly listen for new connection via a thread.
		+ B comes and connects to A as a client, then he also sends an Add Peer Message (type 2) to A. 
		+ C connects to the network via any nodes in the network (in this case C choose B). C would connect to B like B connected to A. Then, C would send an Add Peer message to B. B would forward this Add Peer to other people in the network (ie A). After this, we have a complete network where everyone is connected to every other person. 
		+ D joins, just like C. In the end, A, B, C, D all connected to one another.
		+ Anyone can leave and the network is still connected. Remove Peer message is sent out automatically as user exit by typing **:q**. 

# Instruction
## 1. Compilation:
```
$ rm p2pchat
$ make
```
Thess commands remove the previously compiled p2pchat if there are any and recompile the program.

## 2. Starting the chat:
```
$ ./p2pchat (username)
```

For example:
```
$ ./p2pchat Alice
```
The terminal will print out message such as 
```
>INFO: Alice at address 127.0.0.1 is listening at 55107. 
```

55107 is the **port_number** that Alice is currently listening to. 127.0.0.1 is the **chatroom_IP**. Other node can connect 
to the network via this **port_number** and **chatroom_IP**.

## 3. Joining the chat:
New user can join the chat by connecting to any nodes inside the network. All they need is IP and port of that node.
```
$ ./p2pchat (username)  (chatroom_IP)  (port_number)
```

The **chatroom_IP** and the **port_number** is the IP and port of **a specific node** in the chatroom. 

For example: 
```
$ ./p2pchat Bob localhost 55107 
```

would connect Bob to Alice chat room mentioned above.

## 4. Quit the chat:
To quit the chat, type **:q** into the chat box.



