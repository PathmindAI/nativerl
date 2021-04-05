cmake_minimum_required(VERSION 3.4)
project(nativerl)

# Download and unpack pybind11 at configure time
configure_file(CMakeLists.txt.in pybind11-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
if(result)
  message(FATAL_ERROR "CMake step for pybind11 failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pybind11-download )
if(result)
  message(FATAL_ERROR "Build step for pybind11 failed: ${result}")
endif()

# Add pybind11 directly to our build. This defines
# the pybind11_add_module() function.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/pybind11-src
                 ${CMAKE_CURRENT_BINARY_DIR}/pybind11-build
                 EXCLUDE_FROM_ALL)

# Work around MSVC creating "Debug" and "Release" subdirectories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}")

# Indicate where to find the JNI library produced by JavaCPP, as required by the Python wrapper
set(PLATFORMDIR "${CMAKE_CURRENT_BINARY_DIR}/classes/ai/skymind/nativerl/$ENV{PLATFORM_HOST}")
set(CXXSOURCES "${PLATFORMDIR}/../nativerl.cpp")

# Pick up the include and library directories provided by JavaCPP
string(REPLACE "$ENV{PLATFORM_PATH_SEPARATOR}" ";" PLATFORM_INCLUDEPATH "$ENV{PLATFORM_INCLUDEPATH}")
string(REPLACE "$ENV{PLATFORM_PATH_SEPARATOR}" ";" PLATFORM_LINKPATH "$ENV{PLATFORM_LINKPATH}")
include_directories(${PLATFORM_INCLUDEPATH})
link_directories(${PLATFORMDIR} ${PLATFORM_LINKPATH})

# Compile the Python wrappers against the JNI library
file(GLOB JNILIBS ${PLATFORMDIR}/*)
file(COPY ${JNILIBS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
pybind11_add_module(nativerl ${CXXSOURCES})
target_link_libraries(nativerl PRIVATE jniNativeRL)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Tweak options to handle correctly C++ exceptions thrown from JavaCPP
    target_compile_options(nativerl PRIVATE /EHc-)
endif()
