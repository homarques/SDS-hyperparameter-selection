# Self-adaptive Data Shifting

MATLAB implementation of the paper ["Hyperparameter selection of one-class support vector machine by self-adaptive data shifting"](https://www.sciencedirect.com/science/article/pii/S0031320317303564).

* [Python implementation](https://github.com/bzantium/OCSVM-hyperparameter-selection)
* [Julia implementation](https://github.com/englhardt/SVDD.jl/blob/master/src/init_strategies/strategies_gamma.jl)

## Importing libraries

This implementation is built on the [PrTools](http://prtools.tudelft.nl/) and [dd_tools](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/pattern-recognition-laboratory/data-and-software/dd-tools/) libraries, both libraries can be imported using the MATLAB command ```addpath```:

```addpath('prtools');``` </br>
```addpath('dd_tools');```

## Generating synthetic one-class dataset
```x = gendatb([300, 0]);``` </br>
```data = gendatoc(x, []);```

![Synthetic dataset](/Figs/original.pdf)



## Generating pseudo binary datasets

### SDS
