import os
from setuptools import setup, find_packages
from io import open

packages = ['cogsciabc'] + ['cogsciabc.' + p for p in find_packages('cogsciabc')]

#with open('requirements.txt', 'r') as f:
#    requirements = f.read().splitlines()

setup(
    name='cogsciabc',
    packages=packages,
    version=0.1,
    author='Antti Kangasrääsiö',
    author_email='antti.kangasraasio@iki.fi',
    url='https://github.com/akangasr/cogsciabc',
#    install_requires=requirements,
    description='ABC for cognitive science models',
    license='MIT')
