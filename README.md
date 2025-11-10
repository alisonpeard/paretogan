# Modifications to Pareto GAN

Modifications are made to the original Pareto GAN code which supports [this paper](https://doi.org/10.48550/arXiv.2101.09113).

Pareto GAN shows that a GAN cannot generate data with heavier tails than the latent space, which generally follows a light-tailed distribution (such as Gaussian). This paper ([and many others](https://doi.org/10.48550/arXiv.2504.21438)) make changes to the architecture and training procedures of GANs to improve training on heavy-tailed data.

This repo investigates whether a simpler approach can achieve similar results. Specifically, transforming training data to a Gaussian distribution before training as normal, then transforming back to the original distribution. We call this method `cauchy2gaussian`. If this approach is acceptable, it avoids the need for more complex modifications to GANs, which have generally been developed for Gaussian data.

## Results

<img src="figs/normal_tailhist.png" width="200" /><img src="figs/pareto_tailhist.png" width="200" /><img src="figs/cauchy2gaussian_tailhist.png" width="200" /><figcaption>**Histograms left to right**: `normal`, `pareto`, `cauchy2gaussian`</figcaption>

Result metrics are shown in the table below. The KS statistic measures the maximum distance between the empirical cumulative distribution functions of the generated and real data. Lower is better. The area function computes the area between the log-log complementary cumulative distribution functions (CCDFs). This metric is particularly sensitive to tail behavior in heavy-tailed distributions. Lower is better.

| Experiment     | KS Statistic | Log-Log Area |
|----------------|--------------|---------------|
| `normal`         | 0.0332       | 52.35         |
| `pareto`         | 0.0123       | 1.97          |
| `cauchy2gaussian`| 0.0059      | 2.07         |

Even though the training data is a mixture of Cauchy distributions, we find that fitting another Cauchy distribution to the mixture provides adequate results. This indicates that the method may still be suitable for data composed of a mixture of unknown heavy-tailed distributions, provided the fitted distribution provides a reasonably good approximation. 

# Pareto GAN

## Install dependencies
```
pip install torch numpy matplotlib pandas scipy
```
Note: we recommend installing torch with GPU support

## Run an experiment
```
python exps.py -ds 0 -type normal
python exps.py -ds 0 -type pareto
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
