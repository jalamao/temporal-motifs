# Temporal Event Graph

A python library for analysing temporal networks (graphs).

#### Features:

1. Building the temporal event graph (static representation of a temporal network)
2. Calculating temporal motifs (with arbitrary number of events)
3. Inter-event time distributions
4. Motif distributions
5. Network decompositions into temporal components
6. Saving/Loading functionality

Please cite the following paper when using:

**The Temporal Event Graph**. *Andrew Mellor* (2017)
[ArXiv Link](https://arxiv.org/abs/1706.02128)

### Installation 

To install first pull the latest version from github with:

```bash
git clone git@github.com:empiricalstateofmind/temporal-motifs.git
```

Install requirements with:

```bash
pip install -r requirements.txt
```

Install the package with:

```bash
python setup.py install
```

or add the package folder to your python path.

Run tests with:

```bash
python -m unittest discover ./motifs/tests/
```

If the tests pass, you're good to go!

*Note that this package has been tested on Python 3.5 only*
