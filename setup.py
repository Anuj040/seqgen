import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = (
    setuptools.find_packages(include=["number_generator", "number_generator.utils"]),
)
print(packages)

setuptools.setup(
    name="number_generator",
    version="0.1",
    description="package for generating sequence images from MNIST digits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anuj040/seqgen.git",
    author="Anuj Arora",
    author_email="anujarora920804@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=setuptools.find_packages(
        include=["number_generator", "number_generator.utils"]
    ),
    install_requires=["numpy", "matplotlib", "Pillow", "requests"],
    include_package_data=True,
)
