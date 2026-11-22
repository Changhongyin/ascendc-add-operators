cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(op179089 LANGUAGES ASC CXX)

set(NPU_ARCH "dav-2201" CACHE STRING "Ascend NPU architecture")
add_executable(op179089 op179089.asc)
target_compile_options(op179089 PRIVATE "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>")
target_link_libraries(op179089 PRIVATE ascendcl)
