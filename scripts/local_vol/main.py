"""
Local vol calibration — entry point.

Toggle EXPERIMENT below, or run the example files directly:

    conda run -n TFM_stable python example_paper.py
    conda run -n TFM_stable python example_impl_vol.py [--date YYYY-MM-DD]
"""

# "paper"    → dummy experiment from Lakhal et al. (LM solver)
# "impl_vol" → SPY implied vol surface (L-BFGS-B solver)
EXPERIMENT = "paper"

if __name__ == "__main__":
    if EXPERIMENT == "paper":
        from example_paper import run
    else:
        from example_impl_vol import run
    run()
