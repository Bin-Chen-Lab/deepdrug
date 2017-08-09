# DeepDrug

The overall goal of this project is to use deep-learning methods to correct low quality drug-induced gene expression profiles. The recent pilot project LINCS has generated a large amount of drug-induced gene expression profiles. However, over half of them are dumped during our regular analysis due to bad quality. This work would be very significant if we could rescue a number of these dumped profiles. Potentially, we will be able to use the corrected profiles in drug discovery ([example 1](http://www.gastrojournal.org/article/S0016-5085(17)30264-0/abstract) , [example 2](https://www.nature.com/articles/ncomms16022)).

As a proof of concept, we have a matrix lincs_signatures_cmpd_landmark_all consisting of 66,511 good samples and  10,000 bad samples, and 978 features (gene expression values). The goal is to use this matrix to train the model and correct the bad samples. We use two methods to evaluate our model.
  - Method 1: [Our recent work](https://www.nature.com/articles/ncomms16022) shows the positive correlation between drug efficacy and RGES derived from gene expression profiles. We expect the correlation increases after the correction. Using the uncorrected samples, the correlation is 0.47.
  - Method 2: We have >300 identical treatments shared by bad and good samples. We expect that their profiles are more similar after the correction. We can consider these treatments as replicates. The expected correlation between two profiles from the identical treatment is 0.57. Without correction, the correlation is 0.148

## GAN (Generative adversarial networks)
We use Generator to generate corrected samples from the bad (fake) samples (note not from random distribution) and use Discriminator to discriminate good and bad samples. We add Regularization to reduce variation.

To run GAN, you need download lincs_signatures_cmpd_landmark_all.RData ([here](https://ucsf.box.com/s/7rskmewkk9tm1llxzdd6muwu4nzil96c)) and lincs_signatures_cmpd_landmark_all.npy ([here](https://ucsf.box.com/s/8rrfobdf10eyydgv362045akjn1wo12a)) and  put under data folder.
```sh
python denoise_gan_baseline.py --data_root ../data --save_folder ~/chenlab_v1/ --cuda --lr 0.0005
```
To visualize loss curves
```sh
tensorboard --logdir ~/chenlab_v1/denoise_gan
```
To evaluate results using method 1
```sh
Rscript evaluate_gan.R ~/proj/deepdrug/data ~/chenlab_v1/denoise_gan/2017-07-27-17-04-15_baseline/ 10000
```
To evaluate results using method 2
```sh
Rscript gan_pred_analysis.R ~/proj/deepdrug/data ~/chenlab_v1/denoise_gan/2017-07-28-02-04-49_baseline/ 40000
```

## denoising autoencoders
We use good samples to train denoising autoencoders and consider the decoded samples as the corrected samples.

```sh
python denoise_autoencoder.py --data_root ~/proj/deepdrug/data --save_folder /mnt/denoise_autoencoder --cuda --n_epoch 100

```
To visualize loss curves
```sh
tensorboard --logdir ~/chenlab_v1/denoise_autoencoder
```

To evaluate results using method 1
```sh
Rscript evaluate_autoencoder.R ~/proj/deepdrug/data 
```
To evaluate results using method 2
```sh
Rscript autoencoder_pred_analysis.R ~/proj/deepdrug/data 
```

## results
This task is different from the image analysis where the input (e.g., random distribution) and output (e.g., image) are differentiable. Our input (bad samples) and output (good samples) are very similar. Most of the times, they are context dependent, meaning that the bad sample of one drug may be very similar to a good sample of another drug. However, within one same drug under the same biological condition, the bad and good samples are easy to classify. Therefore, we need to add drug and its condition (dose, time, cell line) into the model.

In G, if we use autoencoder, the output (new gene expression profile) might vary widely, that's why we saw the large variation of correlation between RGES and drug efficacy. The correlation can be up to 0.6, but also down to 0.3, even the loss curve converges.

In D, If we only use gene expression data alone and ask the model to discriminate bad/good samples, the loss is high. But if we label each sample (drug + dose + time + cell) and ask the model to classify them, the loss could be low, and using method 2, the correlation increases to 0.3 from 0.14.

Evaluation method 1: Occasionally, we saw the correlation could be higher than 0.5 (base line 0.47), but given the variation of the autoencoder, we have not developed a model that can consistently outperform the base line. If the output of autoencoder is residual, the correlation barely changed.

Evaluation method 2: The correlation can increase to 0.3 consistently among multiple models.

# tuning experience
* The parameters (e.g., learning rate, G/D iteration ratio, # layers) have some effect on the curves, but little effect on the results. All the curves seem to converge at the same point. Therefore, the evaluation results are very close.
* The input data is relatively critical.
* once it converges, there is little chance the curve will jump up/down if the training set is large.


## To do list:
* Optimize GAN: the output is very close to the input, though the loss curves look great.
* Use standard methods to evaluate results (e.g., collapse, stability)
* Add denoise auto-encoder
* In current LINCS dataset, we are using the consensus signatures merged from multiple signatures. We may go to analyze the raw signatures instead.
* add drug and biological condition data into the model
