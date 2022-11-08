# oakx-grape

🌳 🍇 Grape wrapper for OAK 🌳 🍇

**ALPHA**

## Usage
Macbook users with M1 processor need to do a few extra steps as follows:

 - Download [Anaconda](https://www.anaconda.com/products/distribution).
 - `conda create --name oakx-grape-env python=3.9`
 - `conda activate oakx-grape-env`
 - `pip install poetry`
 - `poetry install`

The steps below are common to everyone.
```
pip install oakx-grape
poetry run runoak -i grape:sqlite:obo:pato relationships --direction both shape
```
### Install NVM + NPM
These [instructions](https://dev.to/ms314006/how-to-install-npm-through-nvm-node-version-manager-5gif) help setup nvm and npm on one's system.

### Install GraphViz and OboGraphViz
- `brew install graphviz`
- `npm install -g obographviz`

## How it works

This plugin implements a grape wrapper. The wrapper in fact wraps two adapters:

1. An adaptor to ensmallen/grape, for performing performance-intensive graph operations
2. An OAK adapter for handling everything else, including lookup by labels, search, predicate filtering, etc

There are two choices of selector:

1. `grape:kgobo:{go,pato,uberon,...}`
2. `grape:OAK-SELECTOR`

with the first pattern, the grape graph is loaded from kgobo, and the oak adapter is loaded from semantic sql

with the second, you can most existing existing OAK adapters:

- sqlite/semsql
- obo
- rdf/owl

Note you CANNOT use a backend like ubegraph or bioportal that relies on API calls

The idea is we will be able to run a notebook like this:

https://github.com/INCATools/ontology-access-kit/blob/main/notebooks/Monarch/PhenIO-Tutorial.ipynb

With the semsim part handled by OAK

## Acknowledgements
 
This [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/README.html) project was developed from the [oakx-plugin-cookiecutter](https://github.com/INCATools/oakx-plugin-cookiecutter) template and will be kept up-to-date using [cruft](https://cruft.github.io/cruft/).