# Pareto GAN

## Install dependencies
```
pip install torch numpy matplotlib pandas scipy
```
Note: we recommend installing torch with GPU support

## Run an experiment
```
python exps.py -ds 0 -type pareto
python exps.py -ds 0 -type normal
python exps.py -ds 0 -type cauchy2gaussian
```

## Options
GAN type (-type): 
 * pareto
 * uniform
 * normal
 * lognormal

Dataset (-ds): 
 * 0: Dual Cauchy

Note: real datasets may not be available anymore. Dual Cauchy is a good "dataset" to illustrate the concept. 
