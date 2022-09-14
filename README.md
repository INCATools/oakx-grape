# oakx-grape

Grape wrapper for OAK

**PRE-ALPHA**

## Usage

```
pip install oakx-grape
runoak -i grape:kgobo.PATO terms
```

NOTE: the plugin mechanism is currently broken, see https://github.com/INCATools/ontology-access-kit/issues/250

For now see https://github.com/INCATools/oakx-grape/blob/main/tests/test_grape_implementation2.py to see exemplar tests

Example:

```bash
poetry run runoak -i grape:PATO terms
```

semsim to come

## TODO

We need to decide on the appropriate wrapping method. Currently Grape ignores all literals. See this issue: https://github.com/AnacletoLAB/ensmallen/issues/175

## Acknowledgements

Created using https://github.com/INCATools/oakx-plugin-cookiecutter by Harshad Hegde
