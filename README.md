# InvDesFlow-AL (The repository is continuously being updated.)
The discovery of novel functional materials with targeted properties remains a fundamental challenge in materials science. In this work, we propose InvDesFlow-AL, an active learning-based generative framework for inverse materials design, which iteratively optimizes material generation towards desired properties.

## Open-Source Models

We have open-sourced these models at [https://zenodo.org/records/15469341](https://zenodo.org/records/15469341).

| File Name                   | Model Description                                                                 |
|----------------------------|------------------------------------------------------------------------------------|
| FormEGNN-weight.hdf5       | Formation energy prediction model for evaluating crystal thermodynamic stability. |
| SuperconGNN-weight.pt      | Superconducting Tc prediction model specialized for ambient-pressure BCN and hydrides. |
| PretrainGenerationModel.ckpt | Pretrained crystal generation model; supports direct generation and fine-tuning for functional materials. |
| CSP-mpts52.ckpt            | Crystal structure prediction model (MP-TS52 dataset).                |
| CSP-mp20.ckpt              | Crystal structure prediction model (MP-20 dataset).                  |
| CSP-perov5.ckpt            | Crystal structure prediction model (perov5 dataset).        |


