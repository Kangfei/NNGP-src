from setuptools import setup
from setuptools import find_packages

setup(name='estimator',
      version='0.1',
      description='NNGP SQL Cardinality estimator',
      author='Kangfei Zhao',
      author_email='kfzhao@se.cuhk.edu.hk',
      license='MIT',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      packages=find_packages())

