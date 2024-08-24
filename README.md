# TUNA

Welcome to TUNA! A user-friendly quantum chemistry program for diatomics.

<br>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Example</title>
  <style>
    .centered-image {
      max-width: 100%;
      height: auto;
    }
    .center-container {
      text-align: center;
    }
  </style>
</head>
<body>
  <div class="center-container">
    <img src="TUNA Logo.svg" alt="Fish swimming through a wavepacket" class="centered-image" />
  </div>
</body>
</html>

## Contents

The repository includes:

* This README file
* The TUNA logo
* The file LICENSE with the MIT license
* The folder TUNA containing Python files
* The installation file pyproject.toml
* The TUNA manual
* A changelog

## Documentation

A copy of the TUNA manual can be found in this repository, and in the directory where the Python files are installed.

## Using TUNA

### Prerequisites
The program requires Python 3.12 or higher and the following packages:

* numpy
* scipy
* matplotlib
* plotly
* scikit-image
* termcolor

### Installation

The simplest way to install TUNA is by running

```
pip install QuantumTUNA
```

### Running

Add the folder where TUNA was installed (on Windows this may be ```/AppData/Local/Programs/Python/Python312/Lib/site-packages/TUNA/```) to PATH, and then run ```tuna --version``` which should print the correct version if TUNA has installed correctly.

