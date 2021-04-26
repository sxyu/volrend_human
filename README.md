# PlenOctree Volume Rendering with LBS

This is a real-time PlenOctree volume renderer written in C++ using OpenGL,
with LBS support.

Based on: https://alexyu.net/plenoctrees

## Building
Please install a recent version of CMake <https://cmake.org>

### Unix-like Systems
```sh
mkdir build && cd build
cmake ..
make -j12
```

- If you do not have CUDA-capable GPU, pass `-DVOLREND_USE_CUDA=OFF` after `cmake ..` to use fragment shader backend, which is also used for the web demo.
  It is slower and does not support mesh-insertion and dependent features such as lumisphere probe.
   **Only CUDA is supported in LBS version right now**

The main real-time PlenOctree rendererer `volrend` and a headless version `volrend_headless` are built. The latter requires CUDA.
There is also an animation maker `volrend_anim`, which I used to make some of the video animations; don't worry about it unless interested.

You should be able to build the project as long as you have GLFW.
On Ubuntu, you will need X-server; you can try
`sudo apt-get install libgl1-mesa-dev libxi-dev libxinerama-dev libxcursor-dev libxrandr-dev libgl1-mesa-dev libglu1-mesa-dev`

### Windows 10
Install Visual Studio (I am using 2019 here). Then
```sh
mkdir build && cd build
cmake .. -G"Visual Studio 16 2019"
cmake --build . --config Release
```
- If you do not have CUDA-capable GPU, pass `-DVOLREND_USE_CUDA=OFF` after `cmake ..` to use fragment shader backend, which is also used for the web demo.
  It is slower and does not support mesh-insertion and dependent features such as lumisphere probe.
   **Only CUDA is supported in LBS version right now**

The main real-time PlenOctree rendererer `volrend` and a headless version `volrend_headless` are built. The latter requires CUDA.
There is also an animation maker `volrend_anim`, which I used to make some of the video animations; don't worry about it unless interested.

### Dependencies
- C++17
- OpenGL
    - any dependencies of GLFW
- CUDA Toolkit, I tried on both 11.0 and 10.2
    - Pass `-DVOLREND_USE_CUDA=OFF` to disable it.
- libpng-dev (only for writing image in headless mode and saving screenshot, optional)

## Run
```sh
./volrend <name>.npz --rig <model_from_render_dir>.npz --goints joint_names.txt
```
See `--help` for flags.

There is an ImGui window which exposes rendering options as well as interactive features mentioned in the paper + video.
For the mesh insertion, only very simple OBJ files (with triangles only) optionally with vertex coloring are supported. 
Some example meshes are in `sample_obj`, and a program to generate SH meshes (just for fun) is in `sample_obj/sh/gen_sh.cpp`.
Please use meshlab to triangulate other mesh.

### Keyboard + Mouse Controls
- Left mouse btn + drag: rotate about camera position
- Right mouse btn + drag: rotate about origin point (can be moved)
- Shift + Left mouse btn + drag: pan camera
- Middle mouse btn + drag: pan camera AND move origin point simultaneously
- Scroll with wheel: move forward/back in z
- WASDQE: move; Shift + WASDQE to move faster
- 123456: preset `world_up` directions, sweep through these keys if scene is using different coordinate system.
- 0: reset the focal length to default, if you messed with it

Lumisphere probe:
- IJKLUO: move the lumisphere probe; Hold shift to move faster


### Offscreen Rendering

The program `volrend_headless` allows you to perform offscreen rendering on a server.

Usage: `./volrend_headless tree.npz -i intrinsics.txt pose1 pose2... [-o out_dir]`

intrinsics.txt should be a 4x4 intrinsics matrix.
pose1, pose2 ... should contain 3x4 or 4x4 c2w pose matrices,
or multiple matrices in a 4Nx4 format.
Add `-r` to use OpenCV camera space instead of NeRF.

Example to render out images:
`./volrend_headless tree.npz --rig model.npz -i intrinsics.txt pose/* -o tree_rend`

The PNG writing is a huge bottleneck. Example to compute the FPS:
`./volrend_headless tree.npz --rig model.npz -i intrinsics.txt pose/*`

