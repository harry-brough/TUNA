from setuptools import setup

setup(
    name='QuantumTuna',
    version='0.4.0',    
    description='A user-friendly quantum chemistry program for diatomics!',
    url='https://github.com/harrybrough1/TUNA',
    author='Harry Brough',
    license='BSD 2-clause',
    packages=['TUNA'],
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      ],

    classifiers=[      
        'Programming Language :: Python :: 3.12',
    ],
)