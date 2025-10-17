from setuptools import setup, find_packages

setup(
    name="scope",
    version="0.1.1",
    packages=find_packages(where='scope'),
    author="Jinhong Deng",
    author_email="jhdengvision@gmail.com",
    description="SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kinredon/SCOPE",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
