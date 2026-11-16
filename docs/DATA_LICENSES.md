RESEARCH DATA ATTRIBUTION AND RELEASE BOUNDARIES

CHASE-DB1
The CHASE_DB1 retinal vessel reference dataset was provided by Kingston
University London in collaboration with St George's, University of London.
Creators/contributors: Sarah Barman, Andreas Hoppe, Paolo Remagnino,
Muhammad Moazam Fraz, Bunyarit Uyyanonvara, Alicja R. Rudnicka and
Christopher G. Owen.
Source: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/
License: Creative Commons Attribution 4.0
https://creativecommons.org/licenses/by/4.0/
Cite: Fraz et al., An Ensemble Classification-Based Approach Applied to
Retinal Blood Vessel Segmentation. IEEE Transactions on Biomedical
Engineering 59 (2012), 2538-2548.
https://doi.org/10.1109/TBME.2012.2205687
The retained raw JPGs and first-observer PNG annotations are copied unchanged.
Preprocessing and model-derived predictions are separately identified in the
artifact manifest. No endorsement by the dataset providers is implied.

HRF
The High-Resolution Fundus Image Database was provided by the Pattern
Recognition Lab and Department of Ophthalmology, FAU Erlangen-Nuremberg,
together with Brno University of Technology and cooperating clinics.
Source: https://www5.cs.fau.de/research/data/fundus-images/
License: Creative Commons Attribution 4.0
https://creativecommons.org/licenses/by/4.0/
Cite: Budai, Bock, Maier, Hornegger and Michelson, Robust Vessel Segmentation
in Fundus Images. International Journal of Biomedical Imaging (2013), 154860.
https://doi.org/10.1155/2013/154860
This release contains the locally retained image and Supervisely annotation
representation. Its exact upstream conversion history remains incompletely
documented; it is not asserted to be a byte-identical official HRF annotation
release. Third-party labelerLogin fields have been removed from the public
annotation JSON metadata; indexed bitmap payloads and other annotation fields
are unchanged. Original and public-file SHA256 values are recorded separately.
Derived arrays retain the original stored experimental values.

DRIVE
Source: https://drive.grand-challenge.org/
Cite: Staal, Abramoff, Niemeijer, Viergever and van Ginneken. Ridge-based
vessel segmentation in color images of the retina. IEEE Transactions on
Medical Imaging 23(4) (2004), 501-509.
https://doi.org/10.1109/TMI.2004.825627
No verified public redistribution license was found for DRIVE source RGB,
annotations or FOV. Those source fields and original containers embedding
them are not included. Public DRIVE model-prediction exports copy retained
NPY prediction-member bytes and omit GT/FOV; they are newly packaged partial
containers, not historically byte-identical original NPZ files. Locally
retained test-label acquisition provenance remains unresolved. The official
current DRIVE site does not supply its test annotations. Obtain required
source materials through their providers and applicable permissions.

GENERAL
These data licenses do not grant a code or model-weight license. No new
license grant for author-owned code/weights is asserted here. Original
research protocol/source records may retain historical workstation paths
needed for recorded SHA256 bindings; path aliases relocate them on installation.
Manuscripts, reviewer responses and private correspondence are not included.
All released metrics/predictions/checkpoints are existing artifacts. Packaging,
checksums and byte-preserving export are not new training, inference or
statistical evaluation. Full historical all-dataset replay still requires the
omitted licensed DRIVE source/GT containers. No complete-clone replay claim
is made.
