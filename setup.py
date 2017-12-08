import os
from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='hansardparser',
    version='0.1.0',
    packages=['hansardparser', 'hansardparser.plenaryparser', 'hansardparser.plenaryparser.models', 'hansardparser.scrapers'],
    # include_package_data=True,
    license='MIT',
    description='A package for parsing Kenya Hansard transcripts',
    long_description=README,
    url='https://github.com/bnjmacdonald/hansardparser',
    author='Bobbie Macdonald',
    author_email='bnjmacdonald@gmail.com',
    install_requires = ['pandas', 'numpy', 'bs4', 'nltk', 'lxml'],
    zip_safe=False
)
