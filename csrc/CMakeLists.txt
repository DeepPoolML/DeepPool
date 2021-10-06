cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(DeepPool)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/nfsnccl")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/cuda")
find_package(Torch REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++14 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

enable_language(CUDA)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS ON)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library("cudasleep"
  CUDASleep.cu
)


# cloning gRPC at configure time.
message(STATUS "Using gRPC via add_subdirectory (FetchContent).")
include(FetchContent)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG        v1.38.1)
FetchContent_MakeAvailable(grpc)

# Since FetchContent uses add_subdirectory under the hood, we can use
# the grpc targets directly from this build.
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(dp_proto "protos/runtime.proto" ABSOLUTE)
get_filename_component(dp_proto_path "${dp_proto}" PATH)

# Generated sources
set(dp_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/runtime.pb.cc")
set(dp_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/runtime.pb.h")
set(dp_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/runtime.grpc.pb.cc")
set(dp_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/runtime.grpc.pb.h")
add_custom_command(
      OUTPUT "${dp_proto_srcs}" "${dp_proto_hdrs}" "${dp_grpc_srcs}" "${dp_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${dp_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${dp_proto}"
      DEPENDS "${dp_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories("$ENV{HOME}/nfsnccl/include")
link_directories("$ENV{HOME}/nfsnccl/lib")

include_directories("$ENV{HOME}/nfsnccl/include")
link_directories("$ENV{HOME}/nfsnccl/lib")

# dp_grpc_proto
add_library(dp_grpc_proto
  ${dp_grpc_srcs}
  ${dp_grpc_hdrs}
  ${dp_proto_srcs}
  ${dp_proto_hdrs})
target_link_libraries(dp_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Linking info for runtime.
add_executable(runtime
  "runtime.cpp" "runnableModule.cpp" "taskManager.cpp" "logger.cpp" 
  "communication.cpp" "rpcService.cpp" "GraphPieces.cpp"
  "CodeLocation.cpp" "Cycles.cpp" "SpinLock.cpp" "ThreadId.cpp" )
target_link_libraries(runtime
  dp_grpc_proto
  cudasleep
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${TORCH_LIBRARIES})

set_property(TARGET runtime PROPERTY CXX_STANDARD 14)
