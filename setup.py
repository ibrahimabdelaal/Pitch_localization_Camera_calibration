from setuptools import setup, find_packages

setup(
    name="soccer_homography",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'matplotlib',
        'Pillow',
        'tqdm',
        'transformers'
    ]
) 