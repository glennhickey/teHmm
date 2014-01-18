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

Genome annotation tracks are specified in files in [BED](http://genome.ucsc.edu/FAQ/FAQformat.html#format1) or [BigWig](http://genome.ucsc.edu/goldenPath/help/bigWig.html) format.  Each track should be in a single file.  In general, BED files should be sorted and not contain any overlapping intervals.  A script is included to do both these operations:

     removeBedOverlaps.py rawBed.bed > cleanBed.bed

Tracks are grouped together along with some metadata in a Track List XML file, which is required for the TE Model. An example Track List is the following:

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

(TODO more testing of multinomial and scaling.  may be some numeric stability issues still)

Training
-----
The TE model is created by training on given track data using the `teHmmTrain.py` script. Two training modes are supported:

* **EM (default)** Model is trained directly from the track data.
* **supervised** (`--supervised` option) Model is trained on given states in a bed file. 

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


