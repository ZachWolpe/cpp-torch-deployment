# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/zachwolpe/miniforge3/bin/cmake

# The command to remove a file.
RM = /Users/zachwolpe/miniforge3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build

# Include any dependencies generated for this target.
include CMakeFiles/torch-cpp-deployment.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/torch-cpp-deployment.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/torch-cpp-deployment.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/torch-cpp-deployment.dir/flags.make

CMakeFiles/torch-cpp-deployment.dir/main.cpp.o: CMakeFiles/torch-cpp-deployment.dir/flags.make
CMakeFiles/torch-cpp-deployment.dir/main.cpp.o: ../main.cpp
CMakeFiles/torch-cpp-deployment.dir/main.cpp.o: CMakeFiles/torch-cpp-deployment.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/torch-cpp-deployment.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/torch-cpp-deployment.dir/main.cpp.o -MF CMakeFiles/torch-cpp-deployment.dir/main.cpp.o.d -o CMakeFiles/torch-cpp-deployment.dir/main.cpp.o -c /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/main.cpp

CMakeFiles/torch-cpp-deployment.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/torch-cpp-deployment.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/main.cpp > CMakeFiles/torch-cpp-deployment.dir/main.cpp.i

CMakeFiles/torch-cpp-deployment.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/torch-cpp-deployment.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/main.cpp -o CMakeFiles/torch-cpp-deployment.dir/main.cpp.s

# Object files for target torch-cpp-deployment
torch__cpp__deployment_OBJECTS = \
"CMakeFiles/torch-cpp-deployment.dir/main.cpp.o"

# External object files for target torch-cpp-deployment
torch__cpp__deployment_EXTERNAL_OBJECTS =

torch-cpp-deployment: CMakeFiles/torch-cpp-deployment.dir/main.cpp.o
torch-cpp-deployment: CMakeFiles/torch-cpp-deployment.dir/build.make
torch-cpp-deployment: /Users/zachwolpe/Desktop/libtorch/lib/libc10.dylib
torch-cpp-deployment: /Users/zachwolpe/Desktop/libtorch/lib/libkineto.a
torch-cpp-deployment: /Users/zachwolpe/Desktop/libtorch/lib/libtorch.dylib
torch-cpp-deployment: /Users/zachwolpe/Desktop/libtorch/lib/libtorch_cpu.dylib
torch-cpp-deployment: /Users/zachwolpe/Desktop/libtorch/lib/libc10.dylib
torch-cpp-deployment: CMakeFiles/torch-cpp-deployment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable torch-cpp-deployment"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/torch-cpp-deployment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/torch-cpp-deployment.dir/build: torch-cpp-deployment
.PHONY : CMakeFiles/torch-cpp-deployment.dir/build

CMakeFiles/torch-cpp-deployment.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/torch-cpp-deployment.dir/cmake_clean.cmake
.PHONY : CMakeFiles/torch-cpp-deployment.dir/clean

CMakeFiles/torch-cpp-deployment.dir/depend:
	cd /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build /Users/zachwolpe/Desktop/cpp-torch-deploy/exe_2_cpp_torchlib/build/CMakeFiles/torch-cpp-deployment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/torch-cpp-deployment.dir/depend

