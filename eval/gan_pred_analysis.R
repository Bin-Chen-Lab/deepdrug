#compare the distribution between real samples and corrected samples
library(Rtsne) # Load package

args <- commandArgs(trailingOnly=T)

data_root <- args[1]
model_root <- args[2]
model_id <- args[3]

pre_path = paste( model_root, "/", "pred/", model_id, ".txt", sep="")

pred = read.csv(pre_path, sep="\t", header =F)

load(paste(data_root, "/", "lincs_signatures_cmpd_landmark_all.RData", sep=""))
good_samples = sample(1:66511, 2000)
fake_samples = sample(66511:76511, 2000)

samples = lincs_signatures[, c(good_samples, fake_samples )]
tsne_out <- Rtsne(t(samples), ) # Run TSNE

plot(tsne_out$Y, col = c(rep(1, 1000), rep(3, 1000 )), pch=20) # Plot the result ,col=cell_tumor, pch=rges_bin

samples_gan = cbind(lincs_signatures[, c(good_samples)], t(pred[fake_samples -66511,]))
tsne_out2 <- Rtsne(t(samples_gan), ) # Run TSNE
plot(tsne_out2$Y, col = c(rep(1, 1000), rep(3, 1000 )), pch=20) # Plot the result ,col=cell_tumor, pch=rges_bin

