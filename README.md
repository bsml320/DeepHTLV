# DeepHTLV
<b> Deep learning for elucidating HTLV-1 integration sites </b> <br>
<br>
<b> Introduction </b> <br>
Human T-lymphotrophic virus 1 (HTLV-1) is the causative agent for adult T-cell leukemia/lymphoma (ATL) and many other human diseases. Accurate and high throughput detection of HTLV-1 integration sites (VISs) across a genome is an important task. Here, we developed DeepHTLV, the first deep learning framework for VIS prediction de novo from sequence, motif discovery and cis-regulatory factors identification. Using our largest curated benchmark integration datasets of HTLV-1, we demonstrated the accuracy of DeepHTLV and its superior performance compared with conventional machine-learning methods by generating more efficient and interpretive feature representations. Through decoding the informative features captured by DeepHTLV, we discovered several important motif clusters with consensus sequences for potential HTLV-1 integration in humans. Furthermore, DeepHTLV revealed interesting cis-regulatory patterns around the VISs. Over 70 DNA transcription factor binding site (TFBSs) were found to have significant association with the detected motifs such as JUN, FOS, and KLF. Literature evidence supported that over half of these TFBSs were involved in either HTLV-1 integration/replication or with HTLV-1 associated diseases, suggesting that DeepHTLV was not only accurate but made functional relevant and meaningful predictions. Our proposed deep learning method and findings will help to elucidate the regulatory mechanisms of VISs in the human genome and benefit the in-depth exploration of their functional effects.
<p>
<b> Installation </b> <br>
Download DeepHTLV <br>
  
```
git clone https://github.com/bsml320/DeepHTLV
``` 
  
  <br><br>
DeepHTLV was implemented in Python version 3.8. The following dependencies are required: numpy, scipy, pandas, h5py, keras version 2.3.1, and tensorflow version 1.15. Install using the following commands <br><br>
  
  ```
 conda create -n DeepHTLV python=3.8
 pip install pandas
 pip install numpy
 pip install scipy
 pip install h5py
 pip install keras==2.3.1
 pip install tensorflow-gpu==1.15
  ``` 
  
  <br><br>
 <b> Data processing </b>
 DeepHTLV was trained and evaluated on our own largest, curated benchmark database of HTLV-1 VISs from the [Viral Integration Site Database (VISDB)] (https://bioinfo.uth.edu/VISDB/index.php/homepage). We retrieved 33,845 positive VIS samples. Each sample consisted of VISs compiled from experimental papers and other database sources. The sites were all indicated with a chromosome, denoted by <i> chr </i>, and an insertion site denoted by a base pair. This information was extracted and to capture surrounding genomic features, we expanded the insertion site by <i> 500bp </i> up and downstream to generate a VIS region of <i> 1kbp </i>. To generate the negative data, the package <i> bedtools </i> is required. You can install it with <br><br>
  ```
  pip install bedtools
  ```
  
  <br><br>
  
  `bedtools random` can be used to generate random sequences. The default number of sequences is 1,000,000 with length of <i> 1 kbp </i>. Seed number was set at 1337. Once this was done, each positive VIS region was expanded by <b> 30 kbp </b> up and downstream to prevent any possible overlaps. This region of <b> 61 kbp </b> was considered a region of exclusion. Using `bedtools intersect -v`, we can find which random sequences do not overlap with the positive exclusion regions. We then removed any redundant sequences using <i> CD-HIT </i> with `cd-hit-est` for within datasets and `cd-hit-est-2d` for between datasets. The similarity threshold (c) was set to 0.9. We wanted to maintain the ratio of positive to negative samples at 1:10 and so the negative sequences were randomly sampled. The final data count was 32,078 positive VISs and 320,780 negative control sequences. <p>
<br>
    <b> Model Construction </b>
    
 
