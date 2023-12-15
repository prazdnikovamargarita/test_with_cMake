# CMakeList.txt : CMake project for test_task, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (test_task "test_task.cpp" "test_task.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET test_task PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
