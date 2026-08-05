# Install AMD ROCm 7.14.0

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
    - [Install ROCm](#install-rocm)
- [Post-installation](#post-installation)
- [Uninstalling](#uninstalling)

# [Install AMD ROCm 7.14.0 #](#install-amd-rocm-rocm-version)

Compare installation methods

ROCm offers four installation methods. If you're unsure, start with package

manager on Linux or tarball on Windows.

| Install method                   | Platform          | Best for                                                                                                                                                    | Install scope                            |
|----------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| Package manager (apt/dnf/zypper) | Linux             | - Traditional Linux installation - OS managed - Auto post-install                                                                                           | System-wide                              |
| pip                              | - Linux - Windows | - Python and ML workflows (PyTorch, JAX) - Auto post-install                                                                                                | Python virtual environment               |
| Tarball                          | - Linux - Windows | - Self-contained, portable setups - Monolithic install (all components included)                                                                            | - System-wide - Custom install directory |
| Runfile                          | Linux             | - Self-contained, guided (GUI or CLI) - Optional offline - Packageless - Single installer for all GPUs - ROCm and amdgpu driver bundled - Auto post-install | - System-wide - Custom install directory |

Use the following selector to choose your installation method for your

supported AMD GPU or APU and operating system. For system requirements and

support information, see the

[Compatibility matrix](../compatibility/compatibility-matrix.html) . To learn more about changes introduced

in ROCm 7.14.0, see the

[Release notes](../about/release-notes.html) .

Note

If your GPU is not listed, it might be community-enabled through TheRock

nightly builds. For more information, see

[TheRock supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) . For

installation guidance, see

[TheRock releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) .

Device family

All

AMD Instinct™

AMD Radeon™

AMD Ryzen™

Installation method

pip

Tarball

## [Prerequisites #](#prerequisites)

## [Installation #](#installation)

### [Install ROCm #](#install-rocm)

Use the following instructions to install ROCm packages on your system.

## [Post-installation #](#post-installation)

After installing ROCm 7.14.0, complete these post-installation steps to

complete your system configuration and validate the installation.

See also

## [Uninstalling #](#uninstalling)

Installation environment

Contents