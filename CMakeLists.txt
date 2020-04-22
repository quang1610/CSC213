cmake_minimum_required(VERSION 3.15)
project(CSC213)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)


include_directories(/usr/local/include)
add_executable(CSC213 main.c)