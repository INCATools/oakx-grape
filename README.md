# oakx-grape

🌳 🍇 Grape wrapper for OAK 🌳 🍇

**ALPHA**

## Usage

```
pip install oakx-grape
runoak -i grape:sqlite:obo:pato relationships --direction both shape
```

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
