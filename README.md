# DeepDrug

The overall goal of this project is to use deep-learning methods to correct low quality drug-induced gene expression profiles. The recent pilot project LINCS has generated a large amount of drug-induced gene expression profiles, however, over half of them are dumped during our regular analysis due to bad quality. This work would be very signficant if we could rescue a number of these dumped profiles. Potentially, we will be able to use the corrected profiles in drug discovery ([example 1](http://www.gastrojournal.org/article/S0016-5085(17)30264-0/abstract) , [example 2](https://www.nature.com/articles/ncomms16022)).

As a proof of concept, we have a matrix lincs_signatures_cmpd_landmark_all consisting of 66,511 good samples and  10,000 bad samples, and 978 features (gene expression values). The goal is to use this matrix to train the model and correct the bad samples. We use two methods to evaluate our model.
  - method 2: [Our recent work](https://www.nature.com/articles/ncomms16022) shows the positive correlation between drug efficacy and RGES derived from gene expression profiles. We expect the correlation increases after the correction.
  - method 2: We have >300 identical treatments shared by bad and good samples. We expect that their profiles are more similar after the correction.

## GAN (Generative adversarial networks)
we use Generator to generate corrected samples from the bad (fake) samples (note not from random distribution) and use Descriminator to descriminate good and bad samples. We add Regularization to reduce variation.

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

To do list:
* Optimize GAN: the output is very close to the input, though the loss curves look great.
* Use standard methods to evaluate results (e.g., collapse, stability)
* Add denoise anto-encoder