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

## Usage

```bash
./ply-to-sog <input.ply> <output.sog> [--bundle] [--sh-iter <N>]
```

### Options

- **`--bundle`**: Creates a single `.sog` file (uncompressed zip archive) containing the manifest and data textures. If omitted, creates a directory structure.
- **`--sh-iter <N>`**: Sets the number of K-Means iterations for Spherical Harmonics (SH) vector clustering (default: 10). Higher values improve SH quality but increase encoding time.


### Example

```bash
# Basic conversion
./ply-to-sog input.ply output.sog --bundle

# High quality SH encoding
./ply-to-sog input.ply output.sog --bundle --sh-iter 50
```

## Acknowledgements

This tool is a C++ port and optimization of sog conversion portion of the [PlayCanvas splat-transform](https://github.com/playcanvas/splat-transform) tool. It implements the same SOG (Spatially Ordered Gaussians) encoding format but is built native integration. Thank you to PlayCanvas for this awesome splat format!
