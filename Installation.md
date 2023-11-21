# Installation
It is highly recommended to follow these steps:
1. Install the package manager [Mamba](https://github.com/conda-forge/miniforge/releases).
Choose the latest installer at the top of the page, click on "show all assets", and download an installer denominated by "Mambaforge-version number-name of your OS.exe", so e.g. "Mambaforge-23.3.1-1-Windows-x86_64.exe" for a Windows 64 bit operating system. Then, execute the installer to install mamba and activate the option "Add Mambaforge to my PATH environment variable".
(⚠ __WARNING__ ⚠: If you have already installed Miniconda, you can install Mamba on top of it but there are compatibility issues with Anaconda. The newest conda version should also work, just replace `mamba` with `conda` in step 2.)
2. Create a new Python environment (replace "name_of_environment" with your desired name) in the command line via
```
mamba create -c conda-forge -n name_of_environment pymc nutpie arviz jupyter matplotlib openpyxl "python=3.10"
```
3. Install PeakPerformance:
- __Recommended__: Clone the PeakPerformance repository, then open the command line, navigate to your local clone, activate a Python environment, and install PeakPerformance via
```
pip install -e .
```
- __Alternatively__: Download the latest Python wheel, then open the command line, navigate to the directory containing the wheel, activate the Python environment created above, and install PeakPerformance via
```
pip install name_of_wheel.whl
```
