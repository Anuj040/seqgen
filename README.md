# MNIST Sequence Generator
![coverage](figs/coverage.svg)\
This is a python implementation of _generate numbers sequence_ and _generate phone numbers_ functions. The code can be used as standalone script or _pip installable package_. Below is a brief about each of these functions.
1. *generate_numbers_sequence()*: Given an Iterable with a sequence of integers, generates a numpy array with a random representation of each integer from **MNIST digits dataset**.
2. *generate_phone_numbers()*: Given an input integer _n_, using *generate_numbers_sequence()* generates _n_ different random phone number (leading digit 0, followed by 10 digits) like sequences.
* For the above, on being provided an _output path_, both functions will save the generated numpy arrays as _*.png_ file(s). 
* The width of the final generated image has to be provided as an input.
* To ascertain the proper working of the above functions, relevant unittests have been implemented.
    * *test_generate_numbers_sequence.py*: Ascertains that the returned values from *generate_numbers_sequence()* are of expected _type, shape, range_.
    * *test_generate_phone_numbers.py*: Checks if the expected number of files have been generated and the files meet the defined specifications.

## Environment
* python == ^3.8 (tested with 3.8, 3.9)
* numpy
* matplotlib
* Pillow
* requests

## Installation Guideline

```sh
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
## Dataset preparation
* Check if the original dataset is available, if not, download from https://data.deepai.org/mnist.zip.
* Get the unsigned bytes from the gzip file for images and labels (train set only for now)
* Extract the images and labels, into the desired shapes and data types (_uint8_).
* Filter the indices for each digit in the dataset (_python dict_).
* In *generate_numbers_sequence()*, convert the data type for images to _float32_, and invert the colors (white: background, black: text).

## Standalone Usage
All commands should be executed from Work Directory
### Sequence Image Generator
To use sequence image generator, please input the following from command line
```python
poetry run python number_generator/api.py sequence --image_width 150 --output_dir outputs --digits 78677
```
**Note**: Digits flag should be one single string of integers
### Phone Number Generator
Please use the following from command line
```python
poetry run python number_generator/api.py phone --num_images 3 --image_width 150 --output_dir outputs
```

## Packaging
### Building a package
* Make a setup.py with required METADATA for the package
* make the package with ```poetry run python setup.py bdist_wheel```　or ```make build```

The above step should provide the following in the work directory.
```sh
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
```python
python -m pip uninstall {path_to}/dist/number_generator-0.1-py3-none-any.whl
```
### Using the package
With the above steps in place, one should be able to import the _number_generator_ module
```python
import number_generator
from number_generator import generate_numbers_sequence, generate_phone_numbers
```

**Attention details for packaging**
* Rename folders and files to meet the naming requirements
* Include __init__.py files

### Pending Imporvements/Features
* Download progress bar for data being downloaded from internet
* Generator function for model training tasks
* Data Augmentation
* spacing

Points
Having Unittests/github actions continuosly helped me catch bugs introduced with various changes.
