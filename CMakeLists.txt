cmake_minimum_required(VERSION 2.8)
project(TEST2)


# Указать путь к папке Face SDK.
# Example: "C:/Program Files/3_24_0"
set(BASE_DIR "<путь_к_папке_Face_SDK>")


# Укажите путь к директории, где находится сборка OpenCV.
# Example: "C:/Users/User/Downloads/opencv/build"
set(OpenCV_DIR "<путь_к_директории_OpenCV>")



find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(TEST2 "TEST2.cpp")

target_include_directories(TEST2 PRIVATE "${BASE_DIR}/include/")
target_include_directories(TEST2 PRIVATE "${BASE_DIR}/examples/cpp/console_arguments_parser/")
target_compile_definitions(TEST2 PRIVATE BASE_DIR="${BASE_DIR}")

set(LIBS ${OpenCV_LIBS})

target_link_libraries(TEST2 ${LIBS}) 

install(TARGETS TEST2 DESTINATION bin)