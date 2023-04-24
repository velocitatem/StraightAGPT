# StraightAGPT

![Beta badge](https://img.shields.io/badge/beta-0.0.1-blue) ![Unsatble badge](https://img.shields.io/badge/stability-unstable-red) ![License badge](https://img.shields.io/badge/license-MIT-green) ![Python badge](https://img.shields.io/badge/python-3.7-blue)


Bringing the Feynman Method to AI. You teach an AI how to solve problems for you.

## Usage
This is how the cli works:

1. Enter task (solve, exit, custom command)
  + Solve: Solve a problem (Provide the question context and goals)
  + Exit: Exit the program
  + Custom command: Enter a custom command for the AI to evaluate


## Installation
1. Clone the repository
2. Install the requirements (`pip install -r requirements.txt`)
3. Run the program (`python student.py`)

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Create a pull request

# Operations for Data Analysis
Here is how the model~user interaction should go:

1. User provides a problem
2. Model identifies the hypotheses
3. Based on the hypotheses and problem, identifies the test statistic
4. Uses tools to compute test-statistic and p-value
5. Interprets the results in context of the problem and hypotheses

We want the model to: Solve complex problems, provide simple calculations
