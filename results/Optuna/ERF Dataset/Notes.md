# Notes

The ERF dataset is the more "canonical" dataset, and attempts were made
to make the Optuna tests more rigorous. The following parameters were 
varied for AE-SINDy.

The AE-SINDy model had two optuna studies. The first study, in `AE-SINDy_AllParams`
had all the relevant parameters varied:
```
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    latent_dim = trial.suggest_int("latent_dim", 1, 4)
    poly_order = trial.suggest_int("poly_order", 2, 3)
    lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
```

The second run, `AE-SINDy_LimParams`, only had `lr`, `batch_size`, and `lambda1_metaweight` varied.
This was done to ensure that `latent_dim=3` and `poly_order=2` were in fact the 
optimal combination to use. The parameters for the limited study were
```
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)
    lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
```
Similar parameters were found by both runs, which is promising for trusting Optuna, and for understanding the network.
I suspect that increasing the dataset size will improve performance because the three networks seem to 
be weak on the same runs.
