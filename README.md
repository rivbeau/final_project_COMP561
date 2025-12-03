To use you will need to download:
-hg19 Chromosomes files given at [](http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz)
-genomic sites of GM12878 cell at [](http://www.cs.mcgill.ca/~blanchem/561/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed.gz)
-experimental bound TF sites at [](http://www.cs.mcgill.ca/~blanchem/561/factorbookMotifPos.txt.gz)
-position weight matrix of all TF mentioned in the previous file at [](http://www.cs.mcgill.ca/~blanchem/561/factorbookMotifPwm.txt.gz)
-DNA shape features files at [ http://rohsdb.cmb.usc.edu](https://rohslab.usc.edu/ftp/hg19/)

Chromosome data should be in a subfolder named ChromFa.

MODELS subfolder include our 3 different models used (SVM, MLP, LR). 
DATA_EXTRACTION subfolder include our data extraction process from the various input files such as Chromosomes, DNA shape features files, genomic sites and PWM of TF.

Our current implementation uses a fix 13 features but that can easily be changed by changing these:

shape_features.py : line 15 remove and include as you wish 
`BW_PATHS = {"MGW":  "hg19.MGW.wig.bw",
    "ProT": "hg19.ProT.wig.bw",
    "Roll": "hg19.Roll.wig.bw",
    "HelT": "hg19.HelT.wig.bw",...}`
LR_final_version.py : 
