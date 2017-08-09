# method 1
check if the corrected samples lead to a better correlation between drug-efficacy and RGES
assume if GAN can generate better samples, the correlation should increase. Using uncorrected samples, the correlation is ~0.47

```sh
Rscript evaluate_gan.R ~/proj/deepdrug/data ~/chenlab_v1/denoise_gan/2017-07-27-17-04-15_baseline/ 10000
```
![Alt Text](method1_base_line_correlation.png){:height="400px" width="400px"}

[[method1_modelx_correlation.pdf]]

<img src="method1_base_line_correlation.png" width="48">

<embed src="method1_modelx_correlation.pdf" width="500" height="375" type='application/pdf'>

# method 2
We have a few hundred identical samples shared by good and bad samples. Identical means they were treated under the same
biological conditions. There profiles should be similar. If GAN is good, it should increase the similarity between two identical samples
```sh
 Rscript gan_pred_analysis.R ~/proj/deepdrug/data ~/chenlab_v1/denoise_gan/2017-07-28-02-04-49_baseline/ 40000
```
