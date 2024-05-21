from setuptools import setup, find_packages

setup(
    name='dataPrep',
    version='0.1.0',
    description='A library to clean data for ML engineers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sarthak and Srishti',
    author_email='sarthakbathla17@gmail.com',
    url='https://github.com/sarthak-star/dataPrep',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
