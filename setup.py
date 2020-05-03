import setuptools

setuptools.setup(
    name="FineTune-pkg-charleschiu",
    version="0.0.1",
    author="Charles Chiu",
    author_email="charleschiu2012@gmail.com",
    description="Use pytorch3d to fine tune Moon position",
    url="https://github.com/charleschiu2012/FineTune.git",
    packages=setuptools.find_packages(),
    install_requires=["torchvision>=0.4", "fvcore"],
    python_requires='>=3.8',
)