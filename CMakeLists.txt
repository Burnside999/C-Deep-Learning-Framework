cmake_minimum_required(VERSION 3.10)
project(CDLF C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB LAYER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/layer/*.c")

set(SOURCES 
    ${LAYER_SOURCES}
    main.c
    network.c
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(CDLF m)