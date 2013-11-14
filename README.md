teHmm
=====

Prototype code for identification of transposable elements in annotated DNA sequence using a HMM.  

Dependencies 
-----
(Make sure PATH and PYTHONPATH are updated accordingly)
* python 2.7
* numpy
* bedtools
* pybedtools
* scikit-learn

Input
-----

All tools take as a parameter a *tracksInfo* xml file that contains some basic information about the input tracks. See tests/data/tracksInfo.xml and tests/data/tracksInfo1.xml for examples. Two track types are currently supported

* Binary BED: Regions covered by the file are presumed to emit symbol 1, whereas all other regions emit 0
* Multinomial BED: The 4th column (name) is used by default for the discrete emitted state.  

Training
-----
The TE model is created by training on given track data using the `teHmmTrain.py` script. Two training modes are supported:

* **EM (default)** Model is trained directly from the track data.
* **supervised** (`--supervised` option) Model is trained on given states in a bed file. It is important to remember that the input file must contain intervals for non-TE regions as well so the model can learn those too.  Simple scripts are provided to help this (`addBedGaps.py` and `addBedCol.py`)

### Supervised Training Example

* `truth.bed` : file where the 4th column represents the true states
* `kmer.bed` : file where 4th column is a number corresponding to kmer coverage
* `ortho.bed` : file where each interval corresponds to an alignment block (but name and other columns not present or meaningful)
* `all.bed` : file with an interval for each chromosome in the genome (must have same number of columns as `truth.bed` for now)

1. Make a training file that contains a default state for everything not in truth.bed: `addGapsToBed.py all.bed truth.bed training.bed`
2. Make a tracks.xml file that contains an entry for `kmer.bed` and `ortho.bed`.  Specifiy their distributions as `multinomial` and `binary`, respectively
3. Crate the model `teHmmTrain.py tracks.xml training.bed te.mod --supervised`

### EM Training Example

Same as above but do not need truth.bed or training.bed: `teHmmTrain.py tracks.xml all.bed te.mod`

