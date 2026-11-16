# Method attribution and retained-source provenance

## clDice and soft skeletons

The structural-metric module attributes hard clDice's centreline precision/sensitivity definition and the soft-skeleton training idea to Shit et al., **clDice—A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**, CVPR 2021, [DOI: 10.1109/CVPR46437.2021.01629](https://doi.org/10.1109/CVPR46437.2021.01629). Its existing source header links the [authors' clDice repository](https://github.com/jocpae/clDice); that attribution is retained. The finite pooling approximation used here is not a topology guarantee.

## Baseline and compatibility modules

The frozen baseline and OOF inputs are the authors' retained executable LU-Net+RA reference. The baseline concept is described by Hernández-Gutiérrez et al., **Retinal Vessel Segmentation Based on a Lightweight U-Net and Reverse Attention**, Mathematics 2025, 13, 2203, [DOI: 10.3390/math13132203](https://doi.org/10.3390/math13132203).

The legacy filename `paired_reproduction/official_model.py` identifies the retained implementation used by these experiments. Its filename alone does not establish that it is an official upstream release or establish an upstream code license. The compatibility modules preserve access to the recorded preprocessing, baseline and fold producers; their presence does not make the new Direct recipes autoencoder methods.

## Licensing

Method citations and provenance notes are acknowledgments, not inferred code-license grants. The authors have not yet selected a license for this repository. Dependencies and source datasets remain subject to their own terms; source images, annotations, model weights and raw prediction archives are not bundled here.
