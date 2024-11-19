# PET Processing Module

Test-bed for Brier Lab PET Processing.

## Installation

In the top-level directory (where `pyproject.toml` exists), we run the following commands in the terminal:

```shell
pip install build  # Ensures that the build package is available
python -m build  # Generates a tarball and a wheel that we can use pip to install
pip install dist/petpal-0.0.2.tar.gz # Installs the package
```

## Generating Documentation

To generate the documentation in HTML using sphinx, assuming we are in the `docs/` directory and that sphinx is
installed:

```shell
make clean
make html 
```

Then, open `doc/build/html/index.html` using any browser or your IDE.