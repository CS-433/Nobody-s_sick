# CS-433 Machine Learning - Project 1

This project is part of the EPFL CS-433 Machine Learning course, providing students with a hands-on opportunity to work through the entire machine learning process on a real-world dataset.

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)

## About
The goal of the project is to apply machine learning techniques to determine the risk of a person in developing CVD based on features of their personal lifestyle factors. The data comes from the Behavioral Risk Factor Surveillance System (BRFSS). Respondents were classified as having coronary heart disease (MICHD) if they reported having been told by a provider they had MICHD or had a heart attack. 

## Getting Started
Navigate to the `run` directory to access and run the code.
Please refer to the list below for an overview of each files' functionality:
- `implementations.py`: Defines machine learning models used for training.
- `helpers.py`: Includes functions for loading and exporting data.
- `helper_funcs.py`: Includes functions for data manipulation and exploration.
- `data_preprocessing.py`: Includes functions for data cleaning, preprocessing, feature selection.
- `plots.py`: Contains functions for visualization.
- `cross_validation.py`: Contains functions for cross validation.
Ensure that the data required for `run.ipynb` is placed in a folder named `data` within the same repository - where the `run.ipynb` file is located. The `data` folder should contain the following CSV files: `x_train.csv`, `x_test.csv`, `y_train.csv`, `train_ids.csv`, and `test_ids.csv`.

### Prerequisites
This project requires Python 3.8 and the following libraries:

- **numpy**
- **seaborn**
- **matplotlib**

## Usage
1. Ensure that the data is placed in a folder named `data` within the same repository - where the `run.ipynb` file is located.
2. Run the `run.ipynb` file.

## Acknowledgments
- [Machine Learning Project 1 GitHub repo](https://github.com/epfml/ML_course/tree/main/projects/project1) - class project repository.
- [2015 Codebook Report](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf) - context-specific information on the survey
- This work results from the collaboration of Ginevra Larroux and Antoine Vincent, EPFL Master's students in Energy Science and Technology.

