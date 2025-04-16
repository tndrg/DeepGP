Multimodal deep learning enhances genomic risk prediction for cardiometabolic diseases in UK Biobank
===
Leveraging Bidirectional Mamba's capacity to capture long-range dependencies across whole-genome SNP data, we propose DeepGP, a deep learning approach to improve genetic prediction for cardiometabolic diseases while providing interpretable insights into risk factors.

## Papar Information
- Authors:  Taiyu Zhu, Upamanyu Ghose, Héctor Climente-González, Joanna M. M. Howson, Sile Hu, Alejo Nevado-Holgado
- Affiliations: University of Oxford, Novo Nordisk Research Centre Oxford
- Preprint: TBA


## Dataset
- UK Biobank Resource under Application Number 53639

## Use
- To train and test the model, run:
  ```
   bash scripts/t2d.sh
  ```



## Directory Hierarchy
```
|—— .gitignore
|—— README.md
|—— args_generator.py
|—— layers
|    |—— Embed.py
|    |—— SelfAttention_Family.py
|    |—— Transformer_EncDec.py
|—— main_genome.py
|—— models
|    |—— BaseModel.py
|    |—— DeepGP.py
|—— scripts
|    |—— t2d.sh
|—— utils.py
```
## Code Details
### Tested Platform
- software
  ```
  Python: 3.10.13
  PyTorch 2.1.1
  PyTorch Lightning 2.0.8
  ```
- hardware
  ```
  CPU: AMD EPYC 7R13 Processor
  GPU: NVIDIA A10 Tensor Core GPU
  ```

## References
We would like to express our gratitude to the following GitHub repositories for their valuable  code and contributions:
- [Mamba](https://github.com/state-spaces/mamba)
- [Vision Mamba](https://github.com/hustvl/Vim)

## License
BSD 3-Clause License

Copyright (c) 2025, University of Oxford. & Novo Nordisk. All rights reserved.

## Citing
Please use the following BibTeX entry.
```
TBA


