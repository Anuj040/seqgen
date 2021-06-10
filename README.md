# MNIST Sequence Generator
![coverage](figs/coverage.svg)

## Environment
* python == ^3.8 (tested with 3.8, 3.9)
* numpy
* matplotlib
* Pillow
* requests

## Installation Guideline

```
git clone git@github.com:Anuj040/seqgen.git [-b <branch_name>]
cd seqgen (Work Directory)

# local environment settings
pyenv local 3.9.1                             # Choose python of choice                                  
python -m pip install poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# In case older version of pip throws installation errors
poetry run python -m pip install --upgrade pip 

# local environment preparation
poetry install

```

## Standalone Usage
```
Script Under preparation
```

Get the unsigned bytes from the gzip file for images and labels (train set only for now)
Extract the images and labels
convert the data types for images to float32, and invert the colors (white: background, black: text)
Filter the indices for each digit in the dataset.
Define a method for retrieving a random image of a given "digit"
define a method for generating n number of random phone-number like sequence images


Implement Unittests: Checks the outputs from generate_numbers_sequence and generate_phone_numbers of number_generator module

## Packaging
### Building a package
* Make a setup.py with required METADATA for the package
* make the package with ```poetry run python setup.py bdist_wheel```

The above step should provide the follwoing in the work directory.
```
    seqgen {work_directory}
    ├── number_generator
    ├── build                  
    │   ├──lib                               
    │   │   ├──
    │   │     :
    │   ├── ...
    │
    ├── dist                  
    │   ├──number_generator-0.1-py3-none-any.whl  
    │
    ├──number_generator.egg-info
    ├──setup.py
    :
    └── ...
```
### Installing the package
```
python -m pip uninstall {path_to}/dist/number_generator-0.1-py3-none-any.whl
```
### Using the package
With the above steps in place, one should be able to import the _number_generator_ module
```
import number_generator
from number_generator import generate_numbers_sequence, generate_phone_numbers
```

** Attention details for packaging
* Rename folders and files to meet the naming requirements
* Include __init__.py files

### Pending Imporvements/Features
* Download progress bar for data being downloaded from internet
* Generator function for model training tasks
* Data Augmentation
* 

Points
Having Unittests/github actions continuosly helped me catch bugs introduced during various changes.
