# Sua warning Cmake version
cmake_minimum_required (VERSION 3.27.9..3.27.9)
project (myproject)
# Bat che do CUDA
enable_language(CUDA)

# Them headers cua CUDA
include_directories ("/usr/local/cuda-12.2/targets/x86_64-linux/include")
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Sua warning --expt-relaxed-constexpr
add_compile_options(--expt-relaxed-constexpr)

# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_executable(demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(demo MiniDNNLib)

# Link them CUDA vao file executable
target_link_libraries(demo cuda)
# Sua loi CUDA
set_property(TARGET demo PROPERTY CUDA_STANDARD 11)