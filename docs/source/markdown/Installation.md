# Installation
It is highly recommended to follow these steps:
1. Install the package manager [Mamba](https://github.com/conda-forge/miniforge/releases).
Choose the latest installer at the top of the page, click on "show all assets", and download an installer denominated by "Mambaforge-version number-name of your OS.exe", so e.g. "Mambaforge-23.3.1-1-Windows-x86_64.exe" for a Windows 64 bit operating system. Then, execute the installer to install mamba and activate the option "Add Mambaforge to my PATH environment variable".

```{caution}
If you have already installed Miniconda, you can install Mamba on top of it but there are compatibility issues with Anaconda.
```

```{note}
The newest conda version should also work, just replace `mamba` with `conda` in step 2.)
```

2. Create a new Python environment in the command line using the provided [`environment.yml`](https://github.com/JuBiotech/peak-performance/blob/main/environment.yml) file from the repo.
   Download `environment.yml` first, then navigate to its location on the command line interface and run the following command:
```
mamba env create -f environment.yml
```
