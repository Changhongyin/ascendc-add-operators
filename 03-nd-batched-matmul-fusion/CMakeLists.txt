cmake_minimum_required(VERSION 3.16)
project(nd_bmm_fused LANGUAGES ASC CXX)

set(NPU_ARCH "dav-2201" CACHE STRING "Ascend NPU architecture (dav-2201 for A2/A3, dav-3510 for A5)")
set(CMAKE_ASC_ARCHITECTURES "${NPU_ARCH}")

add_executable(nd_bmm_fused main.asc)
target_compile_options(nd_bmm_fused PRIVATE -O2 -std=c++17)
target_link_libraries(nd_bmm_fused PRIVATE ascendcl)
