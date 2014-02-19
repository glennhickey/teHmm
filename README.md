teHmm
=====

Prototype code for identification of transposable elements in annotated DNA sequence using a HMM.  

Installation
-----

The following software needs to be installed first.  They can usually be installed via your linux distribution's package manager, from source, MacPorts, or easy_install. (Make sure PATH and PYTHONPATH are updated accordingly)
* [git](http://git-scm.com/downloads)
* [python 2.7](http://www.python.org/getit/)
* [cython](http://docs.cython.org/src/quickstart/install.html)
* [numpy](http://www.scipy.org/install.html)
* [bedtools](https://code.google.com/p/bedtools/downloads/list)
* [pybedtools](http://pythonhosted.org/pybedtools/main.html)
* [scikit-learn](http://scikit-learn.org/stable/install.html#install-official-release)
* [bigWigtoBedGraph for BigWig support](http://hgdownload.cse.ucsc.edu/admin/exe/)

teHmm can then be downloaded and installed as follows:

     git clone git@github.com:glennhickey/teHmm.git
     cd teHmm
     ./setup.sh

It's also a good idea to add teHmm to your PATH and PYTHON path.  If you cloned it in /home/tools, then you would run the following:

     export PATH=/home/tools/teHmm/bin:${PATH}
     export PYTHONPATH=/home/tools/:${PYTHONPATH}


Annotation Tracks
-----

Genome annotation tracks are specified in files in [BED](http://genome.ucsc.edu/FAQ/FAQformat.html#format1),  [BigWig](http://genome.ucsc.edu/goldenPath/help/bigWig.html) or [Fasta](http://en.wikipedia.org/wiki/FASTA_format) format.  Each track should be in a single file.  In general, BED files should be sorted and not contain any overlapping intervals.  A script is included to do both these operations:

     removeBedOverlaps.py rawBed.bed > cleanBed.bed

Chromosome (or contig) names must be consistent within all the track files.  Tracks are grouped together along with some metadata in a Track List XML file, which is required for the TE Model. An example Track List is the following:

     <teModelConfig>
     <track name="15mer" path="15mer-threshold50.bed" distribution="binary"/>
     <track name="chaux" path="chaux.bed" distribution="binary"/>
     </teModelConfig>

The track list file contains a single *teModelConfig* element which in turn contains a list of *track* elements.  Each *track* element must have a (unique) *name* attribute and a *path* attribute.  The *path* is either absolute or relative to where ever you launch the stript from (TODO: probably better to make relative to the xml file). Optional attributes are as follows:
* *distribution* which can take the following values: 
  * *binary*, where bed intervals specify 1 and all other regions are 0
  * *multnomial* where the bed value is read from the *name* column of the bed file. Regions outside bed intervals are assumed to have a default value
  * *sparse_multinomial* same as above except regions outside of intervals are considered unobserved.
* *valCol* 0-based (so name=3) column of bed file to read state from for multinomial distribution
* *scale* Scale values by spefied factor and round them to an integer (useful for binning numeric states)
* *logScale* As above, but first apply log transform.

Training
-----
The TE model is created by training on given track data using the `teHmmTrain.py` script. Two training modes are supported:

* **EM (default)** Model is trained directly from the track data using expectation-maximization (Baum-Welch algorithm for HMMs).
* **supervised** (`--supervised` option) Model is trained on given states in a bed file which represents a known, true annotation. 

### EM Training

The model can be trained from unnanotated (ie true states not known) data using expectation maximization (EM).   The minum information required is the number of states in the model, specifiable with the `--numStates` option.  By default, this will initialize the transition matrix to a flat distribution: the probability from each state to each other state, including itself, is 1/numStates.   The emission probabilities will be randomly assigned, by default.   Options are provided to tune this behaviour:

* `--initTransProbs`  Specify initial transition prbabilities in a text file, where each line represents an adjacency in the transition matrix.  This file has three columns (separated by any combination of tab or space characters): `fromState  toState  transitionProbability`.  Not all edges need to be assigned in this file.  The remaining probabilitiy (the amount required such that the outgoing probabilities of each edge sums to 1) will be divided among the remaining edges.  NOTE:  This option overrides `--numStates`, and only the states specified in this file will appear in the model.  

* `--fixTrans`  Do not learn the transition probabilities: the matrix specified with `--initTransProbs` will be preserved as-is in the output model.

* `--initEmProbs`  Specify initial emission prbabilities in a text file, where each line a single emission probability.  This file has four columns (separated by any combination of tab or space characters): `stateName  trackName  symbol  emissionProbability`.  Not all emissions need to be assigned in this file.  The remaining probabilitiy (the amount required such that the emisison probabilities of each state for each track sums to 1) will be divided among the remaining symbols.  NOTE: It is important that the state, track and symbol names are compatible with the input transition probabilities and annotation tracks.

* `--fixEm`  Do not learn the emission probabilities: the values specified with `--initEmProbs` will be preserved as-is in the output model.

### EM Training Example

This is an example of how to use the options described above to specify a HMM that has the following components:

* **Outside**  Single state representing genomic regions not containing  any TEs
* **LTR Element**  LTR TE that's represented by 3 states:  LeftLTR->InsideLTR->RightLTR
* **LINE Element**  LINE TE that's represented by 3 states:  PolyALine<->InsideLine<->PolyTLine
* **Other1**  Single state free to train other signals
* **Other2**  Single state free to train other signals

In total, there are 9 states in this example.   We begin by specifying the "edges" of the HMM.  This is done by making sure that only valid transitions are initialized to non-zero probabilities.  Transitions that are initialized to 0 can never change via Baum-Welch training.   For our model, we want to insure that  TE states must return to the Outside state before beginning a new element.   This is accomplished by with the following transition matrix, specified with the `--initTransProbs` option:

    Outside  LeftLTR  0.1
	Outside  PolyALine  0.1
	Outside  PolyTLine 0.1
	Outside  Other1  0.1
	Outside  Other2  0.1
	Outside  Outside 0.5

	LeftLTR  InsideLTR  0.5
	LeftLTR  LeftLTR  0.5
	InsideLTR  RightLTR  0.5
	InsideLTR  InsideLTR  0.5
	RightLTR  Outside  0.5
	RightLTR RightLTR  0.5

	PolyALine  InsideLine 0.25
	PolyALine  Outside 0.25
	PolyALine  PolyALine 0.5
	InsideLine  PolyALine 0.25
	InsideLine  PolyTLine 0.25
	InsideLine  InsideLine 0.5
	PolyTLine  InsideLine 0.25
	PolyTLine  Outside 0.25
	PolyTLine  PolyTLine 0.5

	Other1  Outside  0.5
	Other1  Other1  0.5

	Other2  Outside  0.5
	Other2  Other2  0.5

Because the outgoing probabilities of each state above sum to 1, all other transitions (ex LeftLTR->rightLTR) will be set to 0.   Even given this information, it is unlikely that the training algorithm will learn the states we want from the data.  This is because there is not enough structure to the model yet to ensure, say, that the LeftLTR state will really correspond to left termini of TEs, rather than any number of other signals in the data tracks.   To help learn the states we want, we provide some hints in the starting emission parameters.   Supposing we have three tracks, "ltrFinder", "chaux" and "fastaSequence" that we want to use as guides.   One possible way of doing this would be to specify the following emission probabilities with the `--initEmProbs` option:

	LeftLTR  ltrFinder  LTR|left|LTR_TE  0.9
	RightLTR  ltrFinder  LTR|right|LTR_TE  0.9
 	InsideLTR  ltrFinder  inside|-|LTR_TE  0.9
	
 	PolyALine  fastaSequence  A  0.99
	PolyALine  chaux  non-LTR/ATLINE  0.9
	InsideLine  chaux  non-LTR/ATLINE  0.9
 	PolyTLine  fastaSequence  T  0.99
	PolyTLine  chaux  non-LTR/ATLINE  0.9

 	Other1  chaux  non-LTR/ATLINE  0.001
	Other1  ltrFinder  LTR|right|LTR_TE  0.001
	Other1  ltrFinder  LTR|left|LTR_TE  0.001
	Other1  ltrFinder   inside|-|LTR_TE 0.001

 	Other2  chaux  non-LTR/ATLINE  0.001
	Other2  ltrFinder  LTR|right|LTR_TE  0.001
	Other2  ltrFinder  LTR|left|LTR_TE  0.001
	Other2  ltrFinder   inside|-|LTR_TE 0.001

 	Outside  chaux  non-LTR/ATLINE  0.001
	Outside  ltrFinder  LTR|right|LTR_TE  0.001
	Outside  ltrFinder  LTR|left|LTR_TE  0.001
	Outside  ltrFinder   inside|-|LTR_TE 0.001

This is essentially saying that we start training under the assumption that the ltrFinder and chaux tracks have 90% sensitivity and 99.9% specificity.  It is important to note that the Baum-Welch algorithm is still free to change any of these probabilities in any direction.  

### Constructing a Gold Standard for Supervised Training

The Gold Standard BED file for supervised training must satisfy the following constraints:
* Each line must have 4 columns
* The 4th column (name) is used to denote the true state at that interval. 
* Intervals must not overlap and must be sorted
* All bases must be given a state. Bases not covered by the bed file are ignored.  It is important therefore to have intervals for regions that are not TEs as they are still important for the model.  

### Supervised Training Example

Suppose we have a file, `ltrfinder.bed`, that was produced by LTR_FINDER that we want to use as the gold standard for training.  The following steps can be performed to construct a training bed. 

1) Make sure the file is sorted and free of overlaps

     removeBedOverlaps.py ltrfinder.bed > ltrfinder_no.bed

2) LTRFINDER bed names look something like "LTR|left|LTR_TE|5".  Because of the unique identifiers, each interval will be associated with its own state if we don't further filter the model.  So we filter the IDs out:

    cleanLtrFinderID.py ltrfinder_no.bed ltrfinder_clean.bed

Note that this script (see the comments in the script file for more details) will also produce a series of other outputs with, for example, symmetric termini, TSDs removed, etc. These bed files are suitable for training. 

3) Make sure that regions between LTR-TEs get a state:

     addBedGaps.py all.bed ltrfinder_clean.bed > ltrfinder_all.bed

Note that here all.bed is a BED file **(with the same number of columns as ltrfinder_no.bed)** containing a single interval for each chromosome or scaffold in the file.  This file is only used to know the size of the genome and only the coordinates are used.  An example for Alyrata is

     scaffold_1	0	33132539	0		  0		 +
     scaffold_2	0	19320864	0		  0		 +	
     scaffold_3	0	24464547	0		  0		 +
     scaffold_4	0	23328337	0		  0		 +
     scaffold_5	0	21221946	0		  0		 +
     scaffold_6	0	25113588	0		  0		 +
     scaffold_7	0	24649197	0		  0		 +
     scaffold_8	0	22951293	0		  0		 +
     scaffold_9	0	1906741	0		  0		 +

This step must also be applied to any other output from step 2) (ex ltr_finder_clean_sym.bed etc.) if these files are to be used for training.

To train the HMM, run:

     teHmmTrain.py tracks.xml ltrfinder_all.bed ltrfinder.hmm

To view the model parameters, run:

     teHmmView.py ltrfinder.hmm

To obtain TE state predictions for scaffold 1:

First create a BED file, `scaffold1.bed` with the desired coordinates:

     scaffold_1	0	33132539	0		  0		 +

Then run the following to ouput to `predictions.bed`:

     teHmmEval.py tracks.xml ltrfinder.hmm scaffold1.bed --bed predictions.bed

### Overriding Trained Parameters

Sometimes it is desirable to tweek a handful of parameters so that they take on certain values no matter what, while leaving everything else to be trained as normal (via supervised or EM learning, for example).  Two options, `--forceTransProbs` and `--forceEmProbs`, are provided to do this.   They take in text files analogous to those described above as the inputs to  `--initTransProbs` and `--initEmProbs`, respectively.   These options are applied after training, overriding the learned values with those specified in the given files.  Probabilities not specified in these files are left trained as-is, except they are re-normalized so that all distributions sum to 1.   For example, to ensure that the "LTRLEFT" state can only emit "LTerm" in the "LastzTermini" track, specify the path of the following text file with teh `--forceEmProbs` option:

     LTRLEFT  LastzTermini  LTerm  1

This will have the effect of renormalizing all other emissions of LTRLEFT on this track to 0.  




