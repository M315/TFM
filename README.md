# Math Masters Final Project

This is the final project of the Master on Mathematics from UNED.

## TODO

### Readings

- [x] ParameterEstimationBSModel_TFM.pdf
- [x] The_Adjoint_Method_Formulation_for_an_Inverse_Problem_in_the_generalized_Black-Scholes_Model_NgenePieba.pdf
- [x] EggerEnglIP20005a.pdf
- [x] Evans_SDE.pdf
- [x] TFG_jose_manuel_mondejar.pdf

### Need review

- [X] BS formula
- [X] Bergomi - Derman results on local vol
- [] Egger examples presentation
    - [X] Example 1
    - [X] Example 2
    - [] Example surface
    - [X] Empirical rule of 2
    - [X] Price replication
- [] Local vol annex
- [X] UNED format
- [] Miguel's corrections
- [X] Introduction
- [] Conclusions
- [] Code
    - [] examples
    - [] market_data
    - [] opt
    - [] pde
    - [X] utils

## Notes

- Start with Black-Scholes and move to more complicated /  relalistic models
    - Heston model: Volatility is stochastic.
    - Leland model: Adds transaction cost to the model.

- Find techniques to make the identification / estimation more robust for fast changing regimes Volmagedon / Crises.

- 20250618
    - Investigate 3 ways of parameter identification:
        - Inverse PDE
        - Pricing Monte-Carlo
        - Statistical (GARCH)? Can it be used in models beyond classic Black-Scholes?