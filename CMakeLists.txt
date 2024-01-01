cmake_minimum_required (VERSION 3.6)
cmake_policy(SET CMP0104 NEW)

project (myproject LANGUAGES CXX CUDA)
enable_language(CUDA)

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_executable(demo ${DIR_SRCS})
set_property(TARGET demo PROPERTY CUDA_ARCHITECTURES 70)


# 添加链接库
target_link_libraries(demo MiniDNNLib)
