# DeepHTLV
## Deep learning for elucidating HTLV-1 integration sites 
### Introduction
Human T-lymphotrophic virus 1 (HTLV-1) is the causative agent for adult T-cell leukemia/lymphoma (ATL) and many other human diseases. Accurate and high throughput detection of HTLV-1 integration sites (VISs) across the host genomes plays a crucial role in the prevention and treatment of HTLV-1 associated diseases. Here, we developed DeepHTLV, the first deep learning framework for VIS prediction de novo from genome sequence, motif discovery and cis-regulatory factor identification. Using our curated, largest benchmark integration datasets of HTLV-1, we demonstrated the accuracy of DeepHTLV and its superior performance compared with conventional machine-learning methods by generating more efficient and interpretive feature representations. Through decoding the informative features captured by DeepHTLV, we discovered eight representative motif clusters with consensus sequences for potential HTLV-1 integration in humans. Furthermore, DeepHTLV revealed interesting cis-regulatory patterns around the VISs. Over 70 DNA transcription factor binding sites (TFBSs) were identified to have significant association with the detected motifs, including JUN, FOS, and KLF. Literature evidence demonstrated that over half of these TFBSs were involved in either HTLV-1 integration/replication or with HTLV-1 associated diseases, suggesting that DeepHTLV not only was accurate but made functionally relevant and meaningful predictions. Our proposed deep learning method and findings will help to elucidate the regulatory mechanisms .
<p>

### Installation
Download DeepHTLV <br>
  
```
git clone https://github.com/bsml320/DeepHTLV
``` 
  
  <br>
DeepHTLV was implemented in Python version 3.6. The following dependencies are required: numpy, scipy, pandas, h5py, keras version 2.3.1, and tensorflow version 1.15. Install using the following commands <p>
  
  ```
 conda create -n DeepHTLV python=3.6
 pip install pandas
 pip install numpy
 pip install scipy
 pip install h5py
 pip install keras==2.3.1
 pip install tensorflow==1.15.0
 pip install tensorflow-gpu==1.1.0
  ``` 
### Data processing
DeepHTLV was trained and evaluated on our own largest, curated benchmark database of HTLV-1 VISs from the [Viral Integration Site Database (VISDB)](https://bioinfo.uth.edu/VISDB/index.php/homepage). We retrieved 33,845 positive VIS samples. Each sample consisted of VISs compiled from experimental papers and other database sources. The sites were all indicated with a chromosome, denoted by <b>chr</b>, and an insertion site denoted by a base pair. This information was extracted and to capture surrounding genomic features, we expanded the insertion site by <b>500 bp</b> up and downstream to generate a VIS region of <b>1000 bp</b>. To generate the negative data, the package <i>bedtools</i> is required. You can install it with <p> 
  ```
  pip install bedtools
  ```
To convert the data files into a usable form for the model, we have to change the BED files into FASTA files. The FASTA files will then be converted into numpy files. You can either use the python script or the Jupyter notebook. To convert BED files into FASTA files, we need a reference genome. We can download hg19 from the UCSC Genome Browser with the following command.
  ```
  ##download reference file and unzip from UCSC Genome Browser
  wget http://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz
  #unzip the file into FASTA format
  gunzip hg19.fa.gz
  ```  
Once the reference genome has been downloaded, you can run python script with `python data_htlv1_.py` or run the Jupyter notebook `data_htlv1.ipynb` to convert the BED files into a usable numpy file for DeepHTLV. <p>

`bedtools random` can be used to generate random sequences. The default number of sequences is 1,000,000 with length of <b>1 kbp</b>. Seed number was set at 1337. Once this was done, each positive VIS region was expanded by <b>30 kbp</b> up and downstream to prevent any possible overlaps. This region of <b>61 kbp</b> was considered a region of exclusion. Using `bedtools intersect -v`, we can find which random sequences do not overlap with the positive exclusion regions. We then removed any redundant sequences using <i> CD-HIT </i> with `cd-hit-est` for within datasets and `cd-hit-est-2d` for between datasets. The similarity threshold (c) was set to 0.9. We wanted to maintain the ratio of positive to negative samples at 1:10 and so the negative sequences were randomly sampled. The final data count was 31,878 positive VISs and 318,780 negative control sequences. The data was split with 9:1 train to test data using `train_test_split` from the <i>scikit-learn</i> package. <p>

### Model construction
DeepHTLV is a convolutional neural network (CNN) consisting of an input layer, a convolutional-pooling module, an attention mechanism, a dense layer, and an output layer. Primary sequences were one-hot encoded into a matrix where each base pair is represented by a binary vectory. A is represented with (1, 0, 0, 0), C (0, 1, 0,0), G (0, 0, 1, 0), and T (0,0,0,1). A convolutional layer was used to capture important sequence features with a specified kernel size and filter length. The activation function is a rectified linear unit (ReLU). The convolutional layer fed into a max-pooling layer for dimensional reduction and noise reduction. The data was then fed into an attention layer. The attention mechanism highlights important genomic positions from the convolutional layer. When fed an input with <i>b x W x h</i> dimensions, it takes the b column and learns the important feature using a dense representation and softmax function. Once the important features are learned, it assigns a position weight matrix (PWM) for the given column. This is repeated for column <i>b</i> for <i>h</i> times until completion. The PWM information from the attention layer is then integrated with the information from the convolutional operation to find the actual genomic positions of these genomic regions. The final output layer uses a sigmoid or logistic regression activation function, which returns the VIS probability of a given genomic sample. Parameter tuning was performed using <i> keras-tuner </i> using the Hyperband method. <p>
<!--![Figure 1](https://user-images.githubusercontent.com/83188410/165390366-24cc4aa7-fcec-409c-9452-bce2a401d04a.jpg)-->
![Figure1_corrected](https://user-images.githubusercontent.com/83188410/165824071-af11284a-5824-4722-ae6f-a1de02489fcf.png)

### Model performance
Model performance was evaluated with two measures: area under the receiver operating characteristic curve (AUROC) and area under the precision recall curve (AUPRC) values. The AUROC measures the trade-off between the true positive rate (TPR) and the false positive rate (FPR) and ranges from 0.5 to 1, where 0.5 is no better than random chance. Due to the imbalanced nature of our data, we use the AUPR to measure the trade-off between the false positives and false negatives. To improve model performance, we implemented a balanced training strategy. The negative data is sampled without replacement at a 1:1 ratio with the positive data and the model is trained. This step is repeated until all of the negative data has been used in the training process, resulting in 10 different DeepHTLV models. The final output and result are the average of the group. We also compared DeepHTLV with 4 traditional machine learning models: Random Forest (RF), Decision Tree (DT), Logistic Regression (LR), and K-Nearest Neighbors (KNN) using the same training strategy. DeepHTLV is superior to these methods with an AUROC improvement of 2-23% with an average AUROC of 0.75 and average AUPR of 0.73. To elucidate the capability of hierarchical representation by DeepHTLV, we visualized the VISs and non-VISs using UMAP (Uniform Manifold Approximation and Projection) method based on the feature representation at varied network layers. We found the feature representation came to be more discriminative along the network layer hierarchy. More specifically, the feature representations for VISs and non-VISs were mixed at the input layer. As the model continues to train, VISs and non-VISs tend to occur in very distinct regions with efficient feature representation.<p>

![Figure 3 copy](https://github.com/bsml320/DeepHTLV/blob/main/data/Figure%203.jpg)
  
<!-- Traditional machine learning methods <p> -->
<!--<img width="536" alt="machinelearningAUROC" src="https://user-images.githubusercontent.com/83188410/165393167-c6dc89f9-6cdd-4103-8ee8-fe1ab28bb622.png">-->

### Motif clustering
In addition to the improved accuracy and performance, kernels in the convolutional layer can be used to find informative sequence features. Kernels in the CNN will be activated by sequence features that are considered important, and those with high mean maximum activation (MMA) help determine which sequence features are important for identifying a sample with a positive label (considered VIS or not). Any kernels with an MMA above a threshold was extracted and aligned with the primary sequence to generate a position weight matrix (PWM) and MEME format sequence subsequence or motif. Clustering analysis shows that there are 8 consensus motif clusters that are important for guiding virus sequence recognition and integration. <p>
    
![Figure 4](https://user-images.githubusercontent.com/83188410/165395791-5901338b-d2ae-4297-ba6d-f30131167e93.jpg)
    
### Transcription Factor Binding Profile Analysis
To decode and understand more about the <i>cis</i>-regulatory factors about HTLV-1 VISs, we performed further exploratory analysis of these motifs. The top 50 most informative motifs were compared with an curated, experimentally validated open source transcription factor binding site (TFBS) database JASPAR2020. Over 70 of these TFs were shown to have a significant associations with the motifs (p < 0.01). We demonstrate that DeepHTLV makes functionally relevant and biologically meaningful predictions by screening literature to find supporting evidence. Over half of the TFBSs, including Jun, Fos, and KLF, had literature evidence indicating their involvement with HTLV-1 or HTLV-1 associated diseases. <p>
      
![Figure 5](https://user-images.githubusercontent.com/83188410/165396375-611e3fce-86a2-4062-8ce7-742d80d21257.jpg)

<!--<img width="767" alt="table1" src="https://user-images.githubusercontent.com/83188410/165398110-0139fa65-72ab-45dc-a00d-439f845a44f9.png">-->
<!--img width="747" alt="table1" src="https://user-images.githubusercontent.com/83188410/165823963-8518b16b-94ab-4dd5-a7df-a7a4cbd87deb.png">-->
![table1_corrected](https://user-images.githubusercontent.com/83188410/168932955-c3edcf59-db71-433f-b1db-fca6993f3a81.png)

  
### Using DeepHTLV
```
python DeepHTLV.py 
```
### Citation and contact
Cite DeepHTLV.  <br>
For any questions please contact [Haodong Xu](mailto:haodong.xu@uth.tmc.edu?subject=[GitHub]%20Source%20Han%20Sans) or [Johnathan Jia](mailto:jdjia93@gmail.com?subject=[GitHub]%20Source%20Han%20Sans).
