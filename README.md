# InvDesFlow-AL 
The discovery of novel functional materials with targeted properties remains a fundamental challenge in materials science. In this work, we propose InvDesFlow-AL, an active learning-based generative framework for inverse materials design, which iteratively optimizes material generation towards desired properties.
### The model and data of **InvDesFlow-AL** have been released, while the code is still under preparation. Before we finalize and perfect the full version of InvDesFlow-AL, you can already use the current version available at:[https://github.com/xqh19970407/InvDesFlow](https://github.com/xqh19970407/InvDesFlow)

[**[Paper]**](https://arxiv.org/pdf/2505.09203)

![Overview](fig/InvDesFlow-AL.png "Overview")

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

## Open-Source Data

| Dataset                   | Link                                                   | Description                                                                                           |
| ------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| Low-Formation-Materials   | [Zenodo 15222702](https://zenodo.org/records/15222702) | 577 113 DFT‑relaxed crystals from 5 active‑learning rounds (generator + DPA‑2 + FormEGNN).            |
| Low‑Ehull‑Materials       | [Zenodo 15221067](https://zenodo.org/records/15221067) | Large set of low‑formation‑energy, compositionally diverse crystals, emphasizing high‑entropy alloys. |
| Candidate‑Superconductors | [Zenodo 14644273](https://zenodo.org/records/14644273) | Collection of candidate superconducting materials.                                                    |


## 📄 Citation
If you use **InvDesFlow-AL** in your research, please cite our work:
```bibtex
@article{InvDesFlow-AL,
  author = {Xiao-Qi Han and Peng-Jie Guo and Ze-Feng Gao and Hao Sun and Zhong-Yi Lu},
  title = {InvDesFlow-AL: active learning-based workflow for inverse design of functional materials},
  journal = {npj Computational Materials},
  year = {2025},
  volume = {11},
  number = {1},
  pages = {364},
  doi = {10.1038/s41524-025-01830-z},
  url = {https://doi.org/10.1038/s41524-025-01830-z},
  issn = {2057-3960},
  date = {2025/11/24}
}
```
