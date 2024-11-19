# Welcome to the CLOUD-D RF Codebase!

This codebase exists as an implementation of the results of our paper, [CLOUD-D RF: Cloud-based Distributed Radio Frequency Heterogeneous Spectrum Sensing.](https://www.amazon.science/publications/cloud-d-rf-cloud-based-distributed-radio-frequency-heterogeneous-spectrum-sensing)

## Installation and Usage Instructions

To install the codebase, we recommend using `pip`:

`pip install .`

## Generating plots from the paper

1. Generate training, validation, and testing data:

    `python scripts/gen.py`

2. Train individual team models:

    `python scripts/train.py`

3. Train baseline fusion model:

    `python scripts/baseline_fusion.py`

4. Train RL and RFE fusion models:

    `python scripts/rl_rfe_fusion_and_analysis.py`

5. Generate plots:

    `python feature_contribution_analysis.py`
   
    `python generate_confusion_matrices.py`
   
    `python plot_accuracy_vs_nuisance_params.py`


## Contributors
| Name | Role | Title | Email |
| ---- | ---- | ----- | ----- |
| Caleb McIrvin | Developer | PhD Student, Spectrum Dominance Division, Virginia Tech National Security Institute | calm@vt.edu |
| Dylan Green | Developer | Masters Student, Spectrum Dominance Division, Virginia Tech National Security Institute | dylang@vt.edu |
| Alyse M. Jones | Developer | Research Associate, Spectrum Dominance Division, Virginia Tech National Security Institute | alysemjones@vt.edu |
| Maymoonah Toubeh | Developer | Research Assistant Professor, Spectrum Dominance Division, Virginia Tech National Security Institute | may93@vt.edu |
| William 'Chris' Headley | Developer | Associate Director, Spectrum Dominance Division, Virginia Tech National Security Institute | cheadley@vt.edu |

## License
[MIT](https://choosealicense.com/licenses/mit/)
