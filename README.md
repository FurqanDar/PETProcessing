# PET Processing Module

Test-bed for Brier Lab PET Processing.

## Installation

In the top-level directory (where `setup.py` exists), we run the following commands in the terminal:

```shell
python setup.py sdist #Generates a tarball that we can use pip to install
pip install dist/petpal-0.0.1.tar.gz #Installs the package
```

## Generating Documentation

To generate the documentation in HTML using sphinx, assuming we are in the `docs/` directory and that sphinx is
installed:

```shell
make clean
make html 
```

Then, open `doc/build/html/index.html` using any browser or your IDE.