# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/towns/C++/slambook/useSophus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/towns/projects/useSophus/build

# Include any dependencies generated for this target.
include CMakeFiles/useSophus.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/useSophus.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/useSophus.dir/flags.make

CMakeFiles/useSophus.dir/main.cpp.o: CMakeFiles/useSophus.dir/flags.make
CMakeFiles/useSophus.dir/main.cpp.o: /home/towns/C++/slambook/useSophus/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/towns/projects/useSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/useSophus.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/useSophus.dir/main.cpp.o -c /home/towns/C++/slambook/useSophus/main.cpp

CMakeFiles/useSophus.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/useSophus.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/towns/C++/slambook/useSophus/main.cpp > CMakeFiles/useSophus.dir/main.cpp.i

CMakeFiles/useSophus.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/useSophus.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/towns/C++/slambook/useSophus/main.cpp -o CMakeFiles/useSophus.dir/main.cpp.s

CMakeFiles/useSophus.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/useSophus.dir/main.cpp.o.requires

CMakeFiles/useSophus.dir/main.cpp.o.provides: CMakeFiles/useSophus.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/useSophus.dir/build.make CMakeFiles/useSophus.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/useSophus.dir/main.cpp.o.provides

CMakeFiles/useSophus.dir/main.cpp.o.provides.build: CMakeFiles/useSophus.dir/main.cpp.o


# Object files for target useSophus
useSophus_OBJECTS = \
"CMakeFiles/useSophus.dir/main.cpp.o"

# External object files for target useSophus
useSophus_EXTERNAL_OBJECTS =

useSophus: CMakeFiles/useSophus.dir/main.cpp.o
useSophus: CMakeFiles/useSophus.dir/build.make
useSophus: /home/towns/Downloads/Sophus/build/libSophus.so
useSophus: CMakeFiles/useSophus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/towns/projects/useSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable useSophus"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/useSophus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/useSophus.dir/build: useSophus

.PHONY : CMakeFiles/useSophus.dir/build

CMakeFiles/useSophus.dir/requires: CMakeFiles/useSophus.dir/main.cpp.o.requires

.PHONY : CMakeFiles/useSophus.dir/requires

CMakeFiles/useSophus.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/useSophus.dir/cmake_clean.cmake
.PHONY : CMakeFiles/useSophus.dir/clean

CMakeFiles/useSophus.dir/depend:
	cd /home/towns/projects/useSophus/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/towns/C++/slambook/useSophus /home/towns/C++/slambook/useSophus /home/towns/projects/useSophus/build /home/towns/projects/useSophus/build /home/towns/projects/useSophus/build/CMakeFiles/useSophus.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/useSophus.dir/depend
