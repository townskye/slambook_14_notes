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
CMAKE_SOURCE_DIR = /home/towns/C++/slambook/ch8_VO2/directMethod

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/towns/C++/slambook/ch8_VO2/directMethod/build

# Include any dependencies generated for this target.
include CMakeFiles/direct_sparse.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/direct_sparse.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/direct_sparse.dir/flags.make

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o: CMakeFiles/direct_sparse.dir/flags.make
CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o: ../direct_sparse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/towns/C++/slambook/ch8_VO2/directMethod/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o -c /home/towns/C++/slambook/ch8_VO2/directMethod/direct_sparse.cpp

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/direct_sparse.dir/direct_sparse.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/towns/C++/slambook/ch8_VO2/directMethod/direct_sparse.cpp > CMakeFiles/direct_sparse.dir/direct_sparse.cpp.i

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/direct_sparse.dir/direct_sparse.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/towns/C++/slambook/ch8_VO2/directMethod/direct_sparse.cpp -o CMakeFiles/direct_sparse.dir/direct_sparse.cpp.s

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.requires:

.PHONY : CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.requires

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.provides: CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.requires
	$(MAKE) -f CMakeFiles/direct_sparse.dir/build.make CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.provides.build
.PHONY : CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.provides

CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.provides.build: CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o


# Object files for target direct_sparse
direct_sparse_OBJECTS = \
"CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o"

# External object files for target direct_sparse
direct_sparse_EXTERNAL_OBJECTS =

direct_sparse: CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o
direct_sparse: CMakeFiles/direct_sparse.dir/build.make
direct_sparse: /usr/local/lib/libopencv_ml.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_superres.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_stitching.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_viz.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_shape.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_objdetect.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_highgui.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_dnn.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_videostab.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_video.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_calib3d.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_features2d.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_flann.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_photo.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_videoio.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_imgcodecs.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_imgproc.so.3.4.11
direct_sparse: /usr/local/lib/libopencv_core.so.3.4.11
direct_sparse: CMakeFiles/direct_sparse.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/towns/C++/slambook/ch8_VO2/directMethod/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable direct_sparse"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/direct_sparse.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/direct_sparse.dir/build: direct_sparse

.PHONY : CMakeFiles/direct_sparse.dir/build

CMakeFiles/direct_sparse.dir/requires: CMakeFiles/direct_sparse.dir/direct_sparse.cpp.o.requires

.PHONY : CMakeFiles/direct_sparse.dir/requires

CMakeFiles/direct_sparse.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/direct_sparse.dir/cmake_clean.cmake
.PHONY : CMakeFiles/direct_sparse.dir/clean

CMakeFiles/direct_sparse.dir/depend:
	cd /home/towns/C++/slambook/ch8_VO2/directMethod/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/towns/C++/slambook/ch8_VO2/directMethod /home/towns/C++/slambook/ch8_VO2/directMethod /home/towns/C++/slambook/ch8_VO2/directMethod/build /home/towns/C++/slambook/ch8_VO2/directMethod/build /home/towns/C++/slambook/ch8_VO2/directMethod/build/CMakeFiles/direct_sparse.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/direct_sparse.dir/depend

