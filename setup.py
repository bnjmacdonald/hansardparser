import os
from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='hansardparser',
    version='0.2.0',
    packages=['hansardparser', 'hansardparser.plenaryparser', 'hansardparser.plenaryparser.models', 'hansardparser.scrapers'],
    # include_package_data=True,
    license='MIT',
    description='A package for parsing Kenya Hansard transcripts',
    long_description=README,
    url='https://github.com/bnjmacdonald/hansardparser',
    author='Bobbie Macdonald',
    author_email='bnjmacdonald@gmail.com',
    install_requires = [
        'chardet==3.0.4',
        'Flask==1.0.2',
        'google-api-python-client==1.7.8',
        'google-cloud-storage==1.14.0',
        'gunicorn==19.9.0',
        'nltk==3.2.1',
        'numpy==1.16.2',
        'pandas==0.24.2',
        'PyPDF2==1.26.0',
        'requests==2.21.0',
        'tensor2tensor==1.13.2',
        'tensorboard==1.13.1',
        'tensorflow==2.3.1',
        'tensorflow-datasets==1.0.1',
        'tensorflow-estimator==1.13.0',
        'tensorflow-hub==0.4.0',
        'tensorflow-metadata==0.13.0',
        'tensorflow-probability==0.6.0',
        'tqdm==4.31.1',
    ],
    zip_safe=False
)
