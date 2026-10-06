# Installation

## Prerequisites

You will need `uv` to manage the Python environment (installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/)).

!!! note
    On an HPC system, `uv` installs to `~/.local/bin`, so make sure your home directory has enough free space.

## Installing CryoCast

??? warning "Isambard-AI: additional prerequisites required"
    Isambard-AI uses ARM processors and there is currently no `aarch64` wheel for `cf-units`.
    Installing `cf-units` requires a compatible `udunits` installation.
    The CryoCast dev team have compiled `udunits` on Isambard-AI and it can be used by setting the following environment variables:

    ```bash
    export UDUNITS2_XML_PATH=/projects/public/u6iz/shared/udunits/share/udunits/udunits2.xml
    export UDUNITS2_INCDIR=/projects/public/u6iz/shared/udunits/include/
    export UDUNITS2_LIBDIR=/projects/public/u6iz/shared/udunits/lib/
    ```

    Alternatively, if you prefer to compile `udunits` yourself, you can do the following:

    ```bash
    mkdir -p $HOME/software/src
    cd $HOME/software/src

    curl -L -o udunits-2.2.28.tar.gz https://downloads.unidata.ucar.edu/udunits/2.2.28/udunits-2.2.28.tar.gz
    tar -xzf udunits-2.2.28.tar.gz
    cd udunits-2.2.28

    ./configure --prefix=$HOME/software/udunits-2.2.28
    make -j4
    make install

    export UDUNITS2_XML_PATH=$HOME/software/udunits-2.2.28/share/udunits/udunits2.xml
    export UDUNITS2_INCDIR=$HOME/software/udunits-2.2.28/include
    export UDUNITS2_LIBDIR=$HOME/software/udunits-2.2.28/lib
    ```

Clone the repository and install with `uv`:

```bash
git clone git@github.com:alan-turing-institute/icenet-mp.git
cd cryocast
uv sync --managed-python
```
