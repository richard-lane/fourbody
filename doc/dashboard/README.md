# Dashboard
[![Github Actions](https://github.com/richard-lane/fourbody/workflows/CI/badge.svg)](https://github.com/richard-lane/fourbody/actions)
[![codecov](https://codecov.io/gh/richard-lane/fourbody/branch/main/graph/badge.svg?token=BCP1DP2V3L)](https://codecov.io/gh/richard-lane/fourbody)

Project status

## Validation tests
A few scripts run as part of the CI to produce plots.  
For each of these tests, phase space `KKππ` events were generated using the `phasespace` package.  

### Projections
Plot projections of phase space variables.
![projections](/../validation_plots/helicity_phsp.png)

### Phi test
Calculate cos(ϕ) and sin(ϕ); check that their sum of squares = 1 for all events.
![phi](/../validation_plots/sin_cos.png)

### Boost test
Boost particles into the K, π and K-π centre of mass frames;  
check that the appropriate three-momenta are 0.
![boosts](/../validation_plots/boosts.png)

### Correlations
Find correlations between phase space variables; plot a heatmap.
![correlations](/../validation_plots/correlations.png)

## Invariant mass parameterisation
An additional, usually worse, parameterisation using only invariant masses is also provided.
![massproj](/../validation_plots/masses_phsp.png)

![masscorr](/../validation_plots/mass_correlations.png)
