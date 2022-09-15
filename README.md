# oakx-grape

Grape wrapper for OAK

**ALPHA**

## Usage

```
pip install oakx-grape
runoak -i grape:pato terms
```

## How it works

Currently when you use the `grape:foo` selector it will do two things:

1. load foo from the semsql repository as a sqlite database, and create a [SqlImplementation](https://incatools.github.io/ontology-access-kit/implementations/sqldb.html)
2. Load foo from the kgobo repository as an ensmallen graph

Various [OAK interfaces](https://incatools.github.io/ontology-access-kit/interfaces/index.html) are delegated to one of these wrapped backends, or potentially multiplexed

The idea is to use Grape for anything requiring performant graph processing (e.g. semsim, embedding) and delegate everything else to OAK

The idea is we will be able to run a notebook like this:

https://github.com/INCATools/ontology-access-kit/blob/main/notebooks/Monarch/PhenIO-Tutorial.ipynb

With the semsim part handled by OAK

## TODO

We need to decide on the appropriate wrapping method. Currently Grape ignores all literals. See this issue: https://github.com/AnacletoLAB/ensmallen/issues/175

## Acknowledgements

Created using https://github.com/INCATools/oakx-plugin-cookiecutter by Harshad Hegde
