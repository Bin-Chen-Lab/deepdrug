#compare the distribution between real samples and corrected samples
library(Rtsne) # Load package

args <- commandArgs(trailingOnly=T)

data_root <- args[1]

pre_path = paste( data_root, "/", "decoded_data.txt", sep="")

pred = read.csv(pre_path, sep="\t", header =F)

#last 10K
pred = pred[nrow(pred)-999:nrow(pred), ]
load(paste(data_root, "/", "lincs_signatures_cmpd_landmark_all.RData", sep=""))

#correlation analysis
colnames(pred) = rownames(lincs_signatures)
print(dim(pred))
print(dim(lincs_signatures[, 66512:ncol(lincs_signatures)]))
rownames(pred) = colnames(lincs_signatures[, 66512:ncol(lincs_signatures)])

cors_before_gan = NULL
cors_after_gan = NULL

mappings = read.csv("mappings_good_bad_id.csv", stringsAsFactors = F)
mappings$good_id = as.character(mappings$good_id )
mappings$bad_id = as.character(mappings$bad_id)

for (i in 1:nrow(mappings)){
  cors_after_gan = c(cors_after_gan,
                      cor(as.numeric(lincs_signatures[,mappings$good_id[i]]),
                          as.numeric(pred[mappings$bad_id[i],])))
  cors_before_gan = c(cors_before_gan, cor(as.numeric(lincs_signatures[,mappings$good_id[i]]),
                        as.numeric( lincs_signatures[, mappings$bad_id[i]])))

}
head(cors_after_gan)
print(paste("correlation between 'identical' treatment before GAN", mean(cors_before_gan)))
print(paste("correlation between 'identical' treatment after GAN", mean(cors_after_gan)))

stop()

#check distribution
good_samples = sample(1:66511, 2000)
fake_samples = sample(66511:76511, 2000)

samples = lincs_signatures[, c(good_samples, fake_samples )]
tsne_out <- Rtsne(t(samples), ) # Run TSNE

plot(tsne_out$Y, col = c(rep(1, 1000), rep(3, 1000 )), pch=20) # Plot the result ,col=cell_tumor, pch=rges_bin

samples_gan = cbind(lincs_signatures[, c(good_samples)], t(pred[fake_samples -66511,]))
tsne_out2 <- Rtsne(t(samples_gan), ) # Run TSNE
plot(tsne_out2$Y, col = c(rep(1, 1000), rep(3, 1000 )), pch=20) # Plot the result ,col=cell_tumor, pch=rges_bin


  
