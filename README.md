# kernel-builder

<img src="https://github.com/user-attachments/assets/85eb3d80-a163-4c67-a9d2-131b8cf4fe6f" width="500" height="500">


This repo contains a Nix package that can be used to build custom machine learning kernels for PyTorch. The kernels are built using the [PyTorch C++ Frontend](https://pytorch.org/cppdocs/frontend.html) and can be loaded from the Hub with the [kernels](https://github.com/huggingface/kernels)
Python package.

This builder is a core component of the larger kernel build/distribution system.

## 🚀 Quick Start

We provide a Docker image with which you can build a kernel:

```bash
# navigate to the activation directory
cd examples/activation

# then run the following command to build the kernel
docker run --rm \
    -v $(pwd):/kernelcode \
    ghcr.io/huggingface/kernel-builder:latest
```

This will build the kernel and save the output in the `build` directory in
the activation folder.

# 📚 Documentation

- [Writing Hub kernels](./docs/writing-kernels.md)
- [Building kernels with Docker](./docs/docker.md)
- [Building kernels with Nix](./docs/nix.md)
- [Local kernel development](docs/local-dev.md) (IDE integration)
- [Why Nix?](./docs/why-nix.md)

## Credits

The generated CMake build files are based on the vLLM build infrastructure.
