# John Von Neumann Institute|Mini Hackathon 2018

### Problem statement

Given a company profile in the past, develop a prototype to predict the company default level at the end of the year.

### In this repo

This repository contains the prototype for the final challenge of the Mini Hackathon hosted by John Von Neumann Institute.

The team achieved the Second Place for the project.

```bash
./
data # input data for training and testing
output # output data after prediction
references # code pieces and scripts for reference
main.py # script for data exploration, training and testing 
Mini-hackathon II-final round-project-description.pdf # project description
report.pdf # report describes the analysis and methods
```

### Prerequisite

Any environment set up such as:

- python3
- pip

### Run

To reproduce the results and graphs in the report, run `python3 main.py`

### Learning

Further data exploration should be carried out before applying the model to train and predict

Although feature selection has been attempt, **feature extraction** (compute new features from available features) should be explored with more care. 