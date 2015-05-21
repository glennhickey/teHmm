PIATEA: A Probabilistic Integrative Approach for Transposable Element Annotation
=====

The PIATEA project aims to create high-quality Transposable Element (TE) annotations by automatically combining multiple evidence tracks using a multivariate HMM.  This package (teHmm) contains the multivariate HMM implementation together with all scripts necessary to preprocess input (in UCSC annotation formats such as BED, BIGBED, BIGWIG) and analyze the output.   We hope this provides a useful framework for not only improving TE annotaitons, but exploring the relationship TEs and their signals in a variety of data sources beyond, for example, just their alignment to a consensus sequence. This method is described in detail in the following article.  All results and figures therein were generated using scripts from this package. 

Hickey et al. *PIATEA: A Probabilistic Integrative Approach for Transposable Element Annotation*, (nearly submitted), 2015.

We welcome feedback:  Please use the GitHub "issues" (top right) for any comments or problems.  

Installation
-----

Python 2.7, NumPy, Cython and BedTools are required for this package to run.  As it stands now, they must be manually installed first.  Normally this is a straightforward process using `easy_install` for the Python packages and `apt-get` (Linux) or `Mac Ports` (OSX) for everything else.  This is a complete list of the dependencies is as follows (Make sure PATH and PYTHONPATH are updated accordingly):
* [git](http://git-scm.com/downloads)
* [python 2.7](http://www.python.org/getit/)
* [cython 0.19.2](http://docs.cython.org/src/quickstart/install.html)
* [numpy 1.72](http://www.scipy.org/install.html)
* [bedtools 2.17](https://code.google.com/p/bedtools/downloads/list)
* [pybedtools 0.62](http://pythonhosted.org/pybedtools/main.html)
* [bigWigToBedGraph for BigWig support](http://hgdownload.cse.ucsc.edu/admin/exe/)
* [bigBedToBed for BigBed support](http://hgdownload.cse.ucsc.edu/admin/exe/)

The PIATEA teHmm package can then be downloaded and installed as follows:

     git clone https://github.com/glennhickey/teHmm.git
     cd teHmm
     ./setup.sh

It some cases, a firewall or other network issue can prevent cloning via https.   The address above can be changed to git@github.com:glennhickey/teHmm.git to access GitHub via ssh instead. 

It's also a good idea to add teHmm to your PATH and PYTHON path.  If you cloned it in /home/tools, then you would run the following:

     export PATH=/home/tools/teHmm/bin:${PATH}
     export PYTHONPATH=/home/tools/:${PYTHONPATH}

Testing
-

Unit tests can be performed by running `./allTests.py` *from the teHmm/ directory*.  If these don't run successfully it's unlikely any of the examples below will either. 

Overview
-----

These are the steps required to run PIATEA.  Each one is explained in more detail in the examples below, but generally corresponds to running a single script from this package.   

1. User collects evidence tracks BED, BIGBED or BIGWIG format, and writes their paths in a simple XML file.  
2. Tracks are preprocessed to ensure they will be compatible with the emission model
3. Tracks are used to automatically segment the genome (optional)
4. HMM is trained via Supervised, Unsupervised or Combined approach
5. Trained HMM used to compute Viterbi (or MPP) prediction across whole genome (or arbitrary subselection), along with posterior probabilities.
6. Guide track is used to label HMM states (optional)
7. Accuracy computed wrt to other annotation (optional)
8. Heatmap and network diagram generated to visualize model (optional). 

Logging
-

By default, most scripts will not display anything to the screen.  Virtually all executables have logging options (--logDebug, --logInfo, --logWarning, --logCritical) to control the verbosity of messages printed.  The --logFile option can be used to print log messages to a file as well as the screen.  Apart from debugging, logging messages can often give an idea of a tool's progress along with the most time-consuming parts of the computation.  

Temporary Files
-

Some temporary files and directories can will be created by many of the programs in this package.  These will always be created in the directory from which the executable is run.  These files can be left on the drive in the event of an early termination, so it it wise to  check for them periodicalyl and delete them (as they can be quite large).  They will generally contain tempXXXXX (where the Xs signify random alhpa-numeric characters).  The temporary files will be listed in the logging output if set to debug (--logDebug).   

Annotation Tracks
-----

Genome annotation tracks are specified in files in [BED](http://genome.ucsc.edu/FAQ/FAQformat.html#format1),  [BigBed](http://genome.ucsc.edu/FAQ/FAQformat.html#format1.5), ([BigWig](http://genome.ucsc.edu/goldenPath/help/bigWig.html) or [Fasta](http://en.wikipedia.org/wiki/FASTA_format) format.  Each track should be in a single file.  In general, BED files should be sorted and not contain any overlapping intervals (as each track is collapsed to one dimension).  A script is included to do both these operations:

     removeBedOverlaps.py rawBed.bed > cleanBed.bed

Chromosome (or contig) names must be consistent within all the track files.  Tracks are grouped together along with some metadata in a Track List XML file, which is required for the TE Model. An example Track List is the following:

     <teModelConfig>
     <track distribution="multinomial" name="RM-RepeatModeler" path="/tracks/clean/alyrata_repeatmodeler_clean.bed" preprocess="rm"/>
	 <track distribution="gaussian" name="GC" path="/tracks/gc/gcPercentWin20.bw"/>
	 <track distribution="multinomial" name="LTR_FINDER" path="/tracks/ltrfinder/alyrata_ltrfinder.bed" preprocess="ltr_finder"/>
	 <track distribution="binary" name="HelitronScanner" path="/tracks/helitronscanner/alyrata_helitronscanner.bed" valCol="0"/>
     <track distribution="mask" name="TRF" path="/tracks/misc/alyrata_trf.bed"/>
     </teModelConfig>

The track list file contains a single *teModelConfig* element which in turn contains a list of *track* elements.  Each *track* element must have a (unique) *name* attribute and a *path* attribute.  The *path* is either absolute or relative to where ever you launch the stript from. Optional attributes are as follows:
* *distribution* which can take the following values: 
  * *binary*, where bed intervals specify 1 and all other regions are 0.  Useful if it doesn't make sense to have a unique HMM symbol for each BED id (which would be behaviour of Multinomial).
  * *multnomial* (**DEFAULT**) where the bed value is read from the *name* column of the bed file. Regions outside bed intervals are assumed to have a default value
  * *sparse_multinomial* same as above except regions outside of intervals are considered unobserved.
  * *gaussian* where each bed value (read as in *multinomial*) must be numeric, and is assumed to be drawn from a Gaussian distribution.  A numeric *default* value must be specified.
  * *mask* track is treated as a masking track where all intervals it covers are completely ignored by the HMM. 
* *valCol* 0-based (so name=3) column of bed file to read state from for multinomial distribution
* *scale* Scale values by spefied factor and round them to an integer (useful for binning numeric states)
* *logScale* .  Scale values by taking logarithm with the given value as base then rounding to integer.  Since log is undefined for values less or equal to 0, the *shift* attribute (see below) needs to be used to in conjunction should such valuers be present in the data.  Failure to do so will result in an assertion error. 
* *delta*.  When set to "true", the value at each is read as an offset from the value at the preivous position.  If no value was read at the previous position, the previous value is assumed to be 0.  This
operation is performed before scaling.  Note, since deltas can be negative this mode is incompatible with *logScale*.  Using this flag with non-numeric tracks is probably not a great idea.  
* *shift*. Add constant to track data value, and is applied before scaling. Useful in conjunction with *logScale* to process values less than or equal to 0.
* *default*. Default value for regions of genome that are not annotated by track (only applies to *multinomial* distribution).  If not specified, unannotated bases are assigned a special NULL symbol.  For numeric tracks, such as those specified in BigWig format, it will probably most often make sense to set default="0.0" (Warning: if default=0 is used in conjunction with logScale, shift must be set to at least 1)
* *preprocess*. Preprocessor action to be applied to track when running `preprcessTracks.py` script on the XML file.  These flags encompass some commonly-used data types in our pipeline, but are fare from exhaustive.  Users inputting tracks from other sources should make sure, especially in the case of multinomial distributions, that the BED IDs make sense. Acceptable values are:
  * *rm*: Apply RepeatMasker output name cleaning using `cleanRM.py` script.
  * *rmu*: As above but the `--keepUnderscore` option is used.
  * *termini*: Apply `cleanTermini.py`
  * *ltr_finder*: Apply `cleanLtrFinderID.py`
  * *overlap*: Apply `removeBedOverlaps.y`

*Note on BED-12 format* By default, only the start and end coordinate (2nd and 3rd BED) column are used.  This may not be desired behaviour for files in BED-12 format, such as gene annotations with the exons in blocks (columns 10-12).   In this case, please see the `cleanGenes.py` script as a potential preprocessing step to convert a gene annotation into a flattened BED file more suitable for the HMM.

All tracks considered in our paper (and some additional ones) are found in the following lists.   Note that for the various experiments in the paper, we generally used a subset of theses lists (leaving out tracks used for validation or tracks that we found to harbour no signal and so on) that should be apparent from the figures and tables.

* [A. Lyrata](https://github.com/glennhickey/teHmm/blob/master/data/mustang_alyrata_tracks.xml)
* [O. Satival](https://github.com/glennhickey/teHmm/blob/master/data/mustang_rice_tracks.xml)
* [D. Mel.](https://github.com/glennhickey/teHmm/blob/master/data/mustang_dm3_tracks.xml)

Preprocessing
-----

**For Multinomial distributions, it is imperative that the input track file contain at most 255 different values**.  This is because by default, the HMM is compiled to use one byte per value in memory.  This can be a problem since many tools will assign a unique ID to each TE name, resulting in millions of unique values (which would yield a useless emission distribution were the 1byte limit to be increased).  Usually, simple name-munging logic (such as cutting off the ID, or extracting only the superfamily) is enough to fix this.  Some scripts are provided to perform these types of operations on the output of tools that were used in our study.  For example, `cleanRM.py`  will transform BED ID "LTR/Copia|rnd-1_family-250" to just "LTR".  `cleanLtrFinderID.py` would transform "TSD|left|LTR_TE|5" to "TSD|left".   These scripts can be run manually before creating the XML file, or automatically using the *preprocess* flag described above.

**Numeric values generally need to be preprocessed  as well**.  For gaussian distributions, it is greatly beneficial in most cases, to round the data so that it will fit into at most 255 bins.   Some times using much fewer bins helps to smooth out noisy tracks (which can be a major source of problems when training).   This smoothing is accomplished with the scaling parameters in the XML described above.  *These parameters can be automatically computed with* `setTrackScaling.py`.

The two steps above (name munging and scaling), can be performed on all tracks in the XML file automatically with a single call to `preprocessTracks.py`.  The output of calling the script (with default values) to the three examples above is:

* [A. Lyrata](https://github.com/glennhickey/teHmm/blob/master/data/mustang_alyrata_tracks_clean.xml)
* [O. Satival](https://github.com/glennhickey/teHmm/blob/master/data/mustang_rice_tracks_clean.xml)
* [D. Mel.](https://github.com/glennhickey/teHmm/blob/master/data/mustang_dm3_tracks_clean.xml)


Segmenting the Genome
-----

By default, the HMM emits a vector of symbols (one symbol per track) for each *base* of the target genomic region.   Performance can be substantially increased, at least in theory, by pre-segmenting the data so that states are emitted for multi-base blocks.   These blocks should contain a minimal amount of variation across all tracks within them.  A tool, `segmentTracks.py`,  is included to use a simple heuristic to generate a segmentation from some input tracks.  The generated segmentation is itself a BED interval (supporting either fixed or variable length segments), which can be passed as an optional parameter to all HMM tools.  The variable-length segmentation uses a very simple smoothing function to assign segments based on track variation, and the HMM algorithms use a heuristic correction to weight the segments by their lengths.  

HMM algorithms are generally (linear) functions of *N*, the number of observations.   Using a segment size with average length 100 will therefore result in a 100-fold speed-up for these computations.  Segmentation can also play an important role addressing convergence and numeric stability problems in training.   An example of variable length segmentation, followed by fixed length semgentation:

	segmentTracks.py tracks.xml alyrata.bed variable_segments.bed 
    segmentTracks.py tracks.xml alyrata.bed fixed100_segments.bed --thresh 999999 --maxLen100

Note that in the results in the paper, we use a fixed length segmentation for training, and a variable length segmentation for evaluation.  This is something that was arrived at by trial and error, but seemed to perform the best on our tests.  The exact parameters will be presented in the complete example at the end. 

Training
-----
The TE model is created by training on given track data using the `teHmmTrain.py` script. Two training modes are supported:

* **EM (default)** Model is trained directly from the track data using expectation-maximization (Baum-Welch algorithm for HMMs).
* **supervised** (`--supervised` option) Model is trained on given states in a bed file which represents a known, true annotation. 

### Unsupervised (EM) Training

The model can be trained from unnanotated (ie true states not known) data using expectation maximization (EM).   The minum information required is the number of states in the model, specifiable with the `--numStates` option.  By default, this will initialize the transition matrix to a flat distribution: the probability from each state to each other state, including itself, is 1/numStates.   The emission probabilities will be randomly assigned, by default.   Options are provided to tune this behaviour:

* `--initTransProbs`  Specify initial transition prbabilities in a text file, where each line represents an adjacency in the transition matrix.  This file has three columns (separated by any combination of tab or space characters): `fromState  toState  transitionProbability`.  Not all edges need to be assigned in this file.  The remaining probabilitiy (the amount required such that the outgoing probabilities of each edge sums to 1) will be divided among the remaining edges.  NOTE:  This option overrides `--numStates`, and only the states specified in this file will appear in the model.  

* `--fixTrans`  Do not learn the transition probabilities: the matrix specified with `--initTransProbs` will be preserved as-is in the output model.

* `--initEmProbs`  Specify initial emission prbabilities in a text file, where each line a single emission probability.  This file has four columns (separated by any combination of tab or space characters): `stateName  trackName  symbol  emissionProbability`.  Not all emissions need to be assigned in this file.  The remaining probabilitiy (the amount required such that the emisison probabilities of each state for each track sums to 1) will be divided among the remaining symbols.  NOTE: It is important that the state, track and symbol names are compatible with the input transition probabilities and annotation tracks.

* `--fixEm`  Do not learn the emission probabilities: the values specified with `--initEmProbs` will be preserved as-is in the output model.

* `--flatEm`  Initialize emissions to flat distribution (they are randomized by default)

* `--reps`  Perform given number of independent training replicates, then save the model with the highest lieklihood.  This is a strategy to help overcome local minima, ands only makes sense when at least some emission probabilities are randomly initialized (`--fixEm`, `--flatEm` not used, and `--initEmProbs` does not specify every parameter if present)

* `--numThreads`  Used to perform independent replicates in parallel using a pool of a given number of threads.  

### Semi-supervised Training Example

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

This is essentially saying that we start training under the assumption that the ltrFinder and chaux tracks have 90% sensitivity and 99.9% specificity.  It is important to note that the Baum-Welch algorithm is still free to change any of these probabilities in any direction.   Note: the script `fitStateNames.py` can be used to rename the OtherX states in the model's prediction to more meaningul names by comparing them to a different annotation.

*Note* Tracks whose distribution is set to gaussian require a different format when specifying emission parameters.  Ex

	Outside  copyNumber  2.5 1.1

Where 2.5 is the mean of the distribution and 1.1 is the standard deviation.


### Automatic Generation of Initial Distributions

Constructing intitial transition and emission distributions by hand, as in the example above, is tedious and error-prone and, from my experience, not likely to result in a model that it inutitively meaningful after training.   An alternative is to use one or more *initial annotation tracks as guides*.   To this end, the `createStartingModel.py` script is provided.   It takes as input a track (chaux, for example), and can automatically generate some simple initial distributions (`createStartingModel.py --help` for more info).

A similar script, `bootstrapModel`, is provided to generate starting distributions from an existing model.  For instance, a model can be created from a gold standard using supervised learning.  It's parameters, or a subset thereof, can be extracted using `bootstrapModel` to use as a baselines for a second round of unsupervised training. 

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

     teHmmTrain.py tracks.xml ltrfinder_all.bed ltrfinder.hmm --supervised

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

Computing the Viterbi prediction and Posterior Distribution
-----

`teHmmEval.py` contains options to output the posterior probability distribution of a given set of states.  This is the probability, for each observation, that the hidden state is in this set.   The posterior distribution for state1 can be obtained for the above example as follows:

teHmmEval.py tracks.xml ltrfinder.hmm segments.bed --bed predictions.bed --segments --pd posteriorDist.bed --pdStates state1

Using a Guide Track to Automatically Name TE States
-----

If unsupervised training was used, then the state names in the model will be 1,2,3...,k.   The Viterbi output will assign one of these numbers to each base.  The hmm parameters (`teHmmView.py`)  can be manually inspected to determine which type of element each state corresponds to but this is an extremely tedious process.   One way around this is to use the confusion matrix between the HMMs annotation and a given "guide track", to help map HMM states to TE states.  In the paper, we use RepeatMasker as a guide track to classify the HMM states as TE and non-TE.  First, we convert the guide track to be a binary TE/non-TE track (this is not required but it simplifies the problem).

    rm2state.sh repeatMasker.bed > repeatMasker2State.bed

Then we use the confusion matrix with the Viterbi output to label the HMM states.

    fitStateNames.py repeatMasker2State viterbi.bed viterbi_relabeled.bed --fdr 0.75

By default, `fitStateNames.py` uses a greedy algorithm to maximize accuracy.  In practice we find that using the `--fdr` option (used to maximize recall), gives better results in most cases.

Note, it is imperative that the two input BEDs to `fitStateNames.py` cover the exact same regions in the same order.  `addBedGaps.py` is a useful tool to ensure this (by adding NULL states to the guide track).  

Example showing the parameters used to train de-novo models in the paper
-----

These same commands (with different input names, of course) we used to train the de-novo models for *A.lyrata*, rice, and fly in the paper.  The non-denovo models were created using the exact same process except a different guide track was used.  Wall time was about 16 hours using up to 10 cores (a lot of that is really slow Python file I/O)

**INPUT**
* `alyrata.bed`  Entire genome (ie one BED interval per chromosome)
* `repeatmodeler.bed`  RepeatModeler output (guide track).  Preprocessed with `cleanRM.py`
* `tracks.xml`  Original tracks XML file.

Use two types of binning (10 bins for segmentation, and 250 bins for HMM).

    preprocessTracks.py tracks.xml alyrata.bed /trackData segTracks.xml
	preprocessTracks.py tracks.xml alyrata.bed /trackData hmmTracks.xml --numBins 250

Divide the genome into 100kb chunks, and randomly draw 350 chunks for training

    chunkBedRegions.py alyrata.bed 100000 > alyrata_chunks.bed
	sampleBedChunks.py alyrata_chunks.bed 35000000 | sortBed > alyrata_sample.bed
	
Create a variable length segmentation of the training region and a fixed length segmentation of the entire genome.  Note the `--delMask 500` parameter: it specifies how the masking tracks are used.  In this case intervals in the masking tracks < 5000b are ignored and positions on either side of such intervals are considered contiguous.  Mask intervals >= 5000b are cut out, with positions on either side forming endpoints of new scaffolds.    

	segmentTracks.py segTracks.xml alyrata_sample.bed training_segments.bed --thresh 99999 --delMask 5000 --maxLen 100
	segmentTracks.py segTracks.xml alyrata.bed eval_segments.bed --thresh 0  --delMask 5000 --stats segStats.txt --chrom alyrata.bed --proc 10

Train the model.  Fully unsupervised with 25 states, 150 iterations, and 5 random replicates.  Random seed set to 0 to help keep comparisons consistent.  `--segLen 0` turns of segment length correction (as we are using fixed length segments).    

	teHmmTrain.py hmmTracks.xml training_segments.bed hmm.mod --segment training_segments.bed --segLen 0 --iter 150 --numStates 25 --fixStart --numThreads 5 --reps 5 --seed 0 --logInfo 2> train.log &

Compute the Viterbi prediction. Here we use the variable-length segments, and the heuristic correction (`--segLen 100`) to match the training input.  

	teHmmEval.py hmmTracks.xml hmm.mod eval_segments.bed --segment --segLen 100 --bed hmm_pred.bed --bic out.bic  --chrom alyrata.bed --proc 10

Fill in masked gaps < 5000 bases using interpolation.  Masked gaps > 5000 bases are left as holes in the output.

	interpolateMaskedRegions.py hmmTracks.xml alyrata.bed hmm_pred_fit.bed hmm_pred_fit_int.bed --maxLen 5000

Compute the confusion matrix labeling.  Note that we can apply this naming back to the model using `applyStateNames.py` and `fitLog.txt` in order to have the correct labels on the heatmaps etc. 

	addBedGaps.py alyrata.bed repeatmodeler.bed repeatmodeler_gapped.bed
	rm2State.sh repeatmodeler_gapped.bed > repeatmodeler_gapped_2state.bed
	fitStateNames.py repeatmodeler_gapped_2state.bed hmm_pred.bed hmm_pred_fit.bed --fdr 0.75 --tl hmmTracks.xml --tgt TE --logDebug 2> fitLog.txt

Finally, the basewise accuracy results are computed with `accTable.py` and a set of other annotations.  The emission heatmap and transition graph are generated with `teHmmView.py`.

Complete List of Tools Included (contents of /bin)
=====

In general, running any executable with `--help` will print a brief description of the tool and all its input parameters.   Often, larger descriptions and examples can be found at the top of the script files themselves.  A few utilities can be found in /scripts.   

**HMM**

* **teHmmTrain.py** : Create a model
* **teHmmEval.py** : Predict most likely sequence of states of input data given a model
* **teHmmView.py**: Print all parameters of a given model.  Options to generate some figures. 
* **createStartingModel.py** :  Given an input track and a prior confidence value, create transition and probability matrix files (which can be tuned by hand) to pass to teHmmTrain.py
* **bootstrapModel.py** : Create transition and probability matrix files (like above) from an existing model.  Facilliates use of multiple rounds of training (say supervised then unsupervised).
* **interpolateMaskedRegions.py** : If masking tracks are used, they will result in gaps in the HMM predictions.  This script can fill these gaps in as a postprocessing step using some simple heuristics.  

**Parameter Selection**
* **statesVsBic.py** : Compute a table of number of unsupervised states vs Bayesian Information Criterion.  

**Track Name Munging**

* **cleanRM.py** : Remove BED ID suffixes (ex. after pipe or slash) to attempt to map IDs to repeat families
* **cleanLTRFinderID.py**:  Similar to above, but designed to only delete the numeric ID at end of token.  Also produces mappings for symmetric and TSD free state names, as well as removes overlaps using score and length for priority. 
* **cleanTermini.py**:  Transform a bed file representing alignments (where aligned regions share same id) into a bed file where, for each aligned pair, the leftmost region is named LTerm and the rightmost region is named RTerm.
* **setBedCol.py**: Set the entire column of a BED file to a given value.

**Track Processing**

* **addBedColours.py**  : Assign unique colours to BED regions according to their ID
* **addBedGaps.py** : Ensure that every base in the genome is covered by a BED file by filling gaps between regions.  Necessary for supervised training.
* **removeBedOverlaps.py** : Sort the bed file and chop regions up so that each base is covered by at most 1 bed interval.  It's important that bed regions never overlap since the HMM can only emit a single value per track per base.
* **removeBedState.py** : Remove all intervals with given ID.  Useful for removing numeric IDs as grepping them out is much more difficult.
* **fillTermini.py** : Add intervals to cover gaps between left and right termini pairs (ie as generated with cleanTermini.py)
* **chopBedStates.py**: Slice up given intervals in BED according to their ID and some other parameters.
* **addTrackHeader.py** : Set or modify track header of BED file in order to display on Genome Browser.
* **filterPredictions.py** : Simple script to remove some obvious artifacts (ex orphaned ltr and tsds) or tiny predictions out of a BED file.
* **filterBedLengths.py** : Filter BED file based on interval length.
* **filterFastaLengths.py** : Filter FASTA file based in sequence length.
* **filterBedScores.py** : Filter BED file based on interval score.
* **cleanGenes.py**: Convert a BED-12 format gene prediction into suitable input for HMM by explicitly splitting block intervals into introns and exons
* **setScoreFromTrackIntersection.py**: Intersect a intervals in BED file with specified track.  Can be used, for example, to map copy number onto RepeatModeler predictions (given a .wig copy number track).  Note that track binning specified in the XML will be applied internally so numeric values will be rounded.  Also note that the **mode** is used to report the average value across the given intervals.

**Scaling, Binning, Chunking, Segmentation, etc.**

* **setTrackScaling.py** : Compute the best scaling factor (between linear and log) for each track in the XML file for a given number of bases, and annotate the XML file accordingly.  Only applies to numeric tracks.
* **scaleVals.py** : Scale each value of a BED or WIG file.  (Above function better as it automatically computes parameters, and doesn't create any new data files)
* **segmentTracks.py** : Segment the track data into chunks of consistent columns.  These chunks can then be considered atomic units (as opposed to bases) by the model using the --segment option.
* **applyTrackScaling.py** : Write scaled versions of tracks in an XML file using the parameters therein (such as those computed by setTrackScaling.py)
* **chunkBedRegions.py** : Chunk up a BED file into approximately equal-sized chunks.
* **sampleBedChunks.py**: Sample the output of `chunkBedRegions.py` do produce, for example, a random training set. 

**Automatic Preprocessing**

* **preprocessTracks.py**: Given a list of input tracks in XML format, run all preprocessing steps (scaling, name munging, automatic TSD finding) to produce a list of tracks usable by the HMM.

**Alignment**

* **tsdFinder.py** : Use kmer hash to find short exact sequence matches between intervals that flank the left and right side of target BED regions.
* **addTsdTrack.py** : Interface that calls tsdFinder.py in order to add a TSD track to a tracks XML file.

**Simple Statistics**

* **valStats.py** : Compute simple statistics of numeric track data
* **countBedStates.py** : Print number of unique IDs
* **bedStats.py** : Generate statistics in CSV spreadsheet format about interval lengths and scores from a BED-file, broken down by ID.  Note that the scores can be mapped from other tracks with `setScoreFromTrackIntersection.py`, so this script could be used to, for example, analyse copy number statistics.

**Validation and Comparison**

* **teHmmBenchmark.py** : Wrapper to train, evaluate, and compare model on given data
* **compareBedStates.py** : Compute base and interval-level precision and recall of one bed file vis-a-vis another.  Both files must cover exactly the same region.
* **fitStateNames.py** : Assign predicted states names according to how well they match a given annotation.  Useful for generating meaningful names for states that were learned in a completely unsupervised manner. 
* **trackRanking.py** : Iteratively call teHmmBenchmark to rank tracks based on their impact on accuracy

**Misc**
* **trackDump.py** : Write an ASCII matrix of track data given an XML track listing and query BED.
* **extractSingleBaseRepeats.py** : Extract runs of a single nucleotide out of a FASTA file and into a BED file (ie use to extract Ns positions or polyA tail candidates)

Credits
-----

This project was developed by Glenn Hickey in [Professor Mathieu Blanchette's](http://www.mcb.mcgill.ca/~blanchem/) lab under his supervision.  Douglas Hoen, Adrian Platts and Professor Thomas Bureau at McGill contributed valuable input and discussions, and provided much of the input tracks for the *A.Lyrata* genome.

Copyright
-----
Released under the MIT license, see LICENSE.txt and source file headers for more details. 




