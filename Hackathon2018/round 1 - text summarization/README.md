# John Von Neumann Institute - Mini Hackathon 2018

### Problem statement

Given a long article, propose a method to summarize the article automatically. Build a prototype to demonstrate the proposed method.

### In this repo

This repository is a project applied to the Mini Hackathon 2017 at John Von Neumann Institute. Mini Hackathon is an active academic activity at the institute, allowing students opportunity for research and apply learning into practical projects.

```bash
./
DUC2003 # 2003 Document Understanding Conference Dataset
summa # summa package 
main.py # the script to run the implementation
report.pdf # a proposal to the method 
input.txt # a long text file to test the prototype
```

### Prerequisite

Any environment set up such as:

- python3
- pip

### Installation

- Sumeval
  - https://github.com/chakki-works/sumeval
    - pip install sumeval
  - After installation you need to download a language model:
    - python -m spacy download en
- Gensim
  - https://github.com/RaRe-Technologies/gensim
    - The simple way to install gensim is:
      - pip install -U gensim
    - Or, if you have instead downloaded and unzipped the source tar.gz package, you’d run:
      - python setup.py test
      - python setup.py install

### Run

To implement, run:

```bash
python3 main.py
```

### Data

Data used in the experiment is the 2003 DUC (2003 Document Understanding Conference Dataset). The data source contains lots of data in the focus of creating short summaries from long written texts. 

Further information about the data can be found [here](<https://duc.nist.gov/duc2003/tasks.html>).

## Methodology 

Details about the method our team has proposed for the challenge can be found in the `report.pdf` in this repository. 

Our team achieved the first place for this challenge. 

We would like to express our appreciation to the author of the package `summa` for what we have learned and applied in this project. 

Further information about the package can be found [here](<https://github.com/summanlp/textrank>) 

