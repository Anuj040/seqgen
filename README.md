![coverage](figs/coverage.svg)

Get the unsigned bytes from the gzip file for images and labels (train set only for now)
Extract the images and labels
convert the data types for images to float32, and invert the colors (white: background, black: text)
Filter the indices for each digit in the dataset.
Define a method for retrieving a random image of a given "digit"
define a method for generating n number of random phone-number like sequence images


Implement Unittests: Checks the outputs from generate_numbers_sequence and generate_phone_numbers of number_generator module