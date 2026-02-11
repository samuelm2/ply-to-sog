# ply-to-sog

A converter tool to transform PLY files into SOG format.

## Dependencies

- **Eigen3**: Required for linear algebra operations.
- **LibWebP**: Required for image compression.
- **OpenMP**: Required for parallel processing.
- **CMake**: Build system (version 3.14+).
- **C++17 Compiler**: Required for filesystem support (e.g., GCC 8+ or Clang 6+).

## Install Dependencies (Ubuntu)

To install all required dependencies on Ubuntu, run:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libeigen3-dev libwebp-dev pkg-config
```

## Build Instructions

1.  Clone the repository:
    ```bash
    git clone https://github.com/samuelm2/ply-to-sog.git
    cd ply-to-sog
    ```

2.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```

3.  Configure and build:
    ```bash
    cmake ..
    make
    ```

4.  Run the converter:
    ```bash
    ./ply-to-sog <input_ply> <output_sog>
    ```
