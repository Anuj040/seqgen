"""common API for low level generator functions (numbers sequence & phone nummbers)"""

import argparse

# pylint: disable = import-error
from num_gen import generate_numbers_sequence, generate_phone_numbers

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="API for passing args to low level functions."
)
parser.add_argument(
    "command",
    metavar="<command>",
    help="function to execute. 'sequence' for generating an \
    image array of sequence of numbers or 'phone' for generating \
    a given number of images in phone-number like pattern",
)
parser.add_argument(
    "-n",
    "--num_images",
    required=False,
    metavar="N",
    type=int,
    default=2,
    help="Number of phone number sequence to be generated",
)
parser.add_argument(
    "-iw",
    "--image_width",
    required=True,
    metavar="N",
    type=int,
    default=100,
    help="Width for the final image",
)
parser.add_argument(
    "-od",
    "--output_dir",
    required=False,
    metavar="path/for/output/images",
    type=str,
    default="outputs",
    help="Directory for output images.",
)
parser.add_argument(
    "-d",
    "--digits",
    required=False,
    type=tuple,
    help="Digits for sequence eg 3629",
)
args = parser.parse_args()


def main():
    """control method for executing one of the generator functions"""

    assert args.command in [
        "phone",
        "sequence",
    ], f"'{args.command}' is not a supported mode. Please chose one of 'phone' or 'sequence'"

    if args.command == "sequence":
        assert (
            args.digits is not None
        ), "Please provide a sequence to be converted to an imge eg. --digits=3629"

    if args.command == "phone":
        generate_phone_numbers(
            num_images=args.num_images,
            image_width=args.image_width,
            output_path=args.output_dir,
        )
    elif args.command == "sequence":
        # Convert into a iterable of ints
        digits = (int(digit) for digit in args.digits)
        generate_numbers_sequence(
            digits=digits,
            image_width=args.image_width,
            output_path=args.output_dir,
        )


if __name__ == "__main__":
    main()
