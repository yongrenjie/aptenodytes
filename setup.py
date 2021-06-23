from setuptools import setup, find_packages

setup(
    name="aptenodytes",
    version="0.0.1",
    author="Jonathan Yong",
    author_email="yongrenjie@gmail.com",
    description=("Personal code which extends the features in the penguins"
                 " package (see https://github.com/yongrenjie/penguins);"
                 " intended for use in my PhD project."),
    url="https://github.com/yongrenjie/aptenodytes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "penguins"
    ]

)
