# PCOT

This is the prototype of the Pancam Operations Toolkit. 

## Installation
PCOT is a Python program (and library) with a number of dependencies:

* Python >3.8
* PyQt
* OpenCV
* numpy
* scikit-image
* pyperclip
* matplotlib

Installation has been tested on Windows 10 and Ubuntu 20.04.

### Anaconda
I find the best way to manage Python versions and PCOT's dependencies is to use Anaconda.
If you wish to use it, install Anaconda from one of the below links before proceeding:
* Windows: https://docs.anaconda.com/anaconda/install/windows/
* Linux: https://docs.anaconda.com/anaconda/install/linux/
* MacOS: https://docs.anaconda.com/anaconda/install/mac-os/ (untested)

#### Opening Anaconda's shell on different OSs
* **Windows:** Open the **Anaconda PowerShell Prompt** application, which will have been installed when you installed Anaconda.
* **Linux and MacOS**: just open a Bash shell 

### Obtaining PCOT
For both Windows and Ubuntu this is the obvious first step. This can be done by
either downloading the archive from Github and extracting it into a new directory,
or cloning the repository. In both cases, the top level directory should be called
PCOT (this isn't really mandatory but makes the instructions below simpler).
The best way to download is this:

* Make sure you have a Github account and membership of the AU-ExoMars group.
* Open an Anaconda shell window (see above) if applicable, else open a regular shell.
* If you have an SSH key set up for GitHub, type this command into the shell:
```shell
git clone git@github.com:AU-ExoMars/PCOT.git
```
* Otherwise type this:
```shell
git clone https://github.com/AU-ExoMars/PCOT.git
```
* You should now have a PCOT directory which will contain this file (as README.md)
and quite a few others.

### Installing PCOT with Anaconda
Assuming you have successfully installed Anaconda and cloned or downloaded PCOT as above:
* Open an Anaconda shell (see above).
* **cd** to the PCOT directory (which contains this file).
* Run the command `conda create -n pcot python=3.8 poetry`. This will create an environment called **pcot**, and will take some time.
* Activate the environment with `conda activate pcot`.
* Install PCOT into the environment with `poetry install`.
* You should now be able to run `pcot` to start the application.

### Installing PCOT without Anaconda
Assuming you have an appropriate version of Python (>=3.8) installed:
* [Install Poetry](https://python-poetry.org/docs/#installation). 
* **cd** to the PCOT directory (which contains this file).
* Run the command `poetry install`. This will create a new virtual environment and install PCOT along with its dependencies into it.
* You should now be able to run the command `poetry run pcot` to start the application.

If you have multiple Pythons installed you can [let Poetry know](https://python-poetry.org/docs/managing-environments/) which version to use.

## Running PCOT
### With Anaconda
Open an Anaconda shell and run the following commands (assuming you installed PCOT into your home directory):
```shell
cd PCOT
conda activate pcot
pcot
```
### Without Anaconda
Open a shell and run the following commands (assuming you installed PCOT into your home directory):
```shell
cd ~/PCOT
poetry run pcot
```





## Running PCOT inside Pycharm
These instructions apply to Anaconda installations.

* First set up the Conda environment and interpreter:
    * Open PyCharm and open the PCOT directory as an existing project.
    * Open **Settings..** (Ctrl+Alt+S)
    * Select **Project:PCOT / Python Interpreter**
    * Select the cogwheel to the right of the Python Interpreter dropdown and then select  **Add**.
    * Select **Conda Environment**.
    * Select **Existing Environment**.
    * Select the environment: it should be something like **anaconda3/envs/pcot/bin/python**.
    * Select **OK**.
* Now set up the run configuration:
    * Select **Edit Configurations** from the configurations drop down in the menu bar
    * Add a new configuration (the + symbol)
    * Set **Script Path** to **PCOT/src/pcot/__main__.py**
    * Make sure the interpreter is something like **Project Default (Python 3.8 (pcot))**, i.e. the Python interpreter of the pcot environment.
* You should now be able to run and debug PCOT.

## Environment variables
It's a good idea, but not mandatory, to set the environment variable
**PCOTUSER** to a string of the form **name \<email\>**. For example,
in Linux I have added the following to my **.bashrc** file:
```
export PCOT_USER="Jim Finnis <jcf12@aber.ac.uk>"
```
This data is added to all saved PCOT graphs. If the environment variable
is not set, the username returned by Python's getpass module is used
(e.g. 'jcf12').
