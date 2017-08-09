################
#para
args <- commandArgs(trailingOnly=T)

data_root <- args[1] #data folder
model_root <- args[2] #model folder
model_id <- args[3]
##################


cmd = paste("python ../denoise_gan_pred_label_case.py --cuda  --data_root", data_root, "--model_root", model_root, "--model_id", model_id, sep=" ")
print(cmd)
system(cmd)

pre_path = paste(model_root, "/", "pred/", model_id, ".txt", sep="")

while(!file.exists(pre_path)){
  Sys.sleep(30)
}

split = 1498 
pred = read.csv(paste( model_root, "/", "pred/", model_id, ".txt", sep=""), sep="\t", header =F)
print(dim(pred))
#

#assign col/row names
load(paste(data_root, "/", "lincs_signatures_cmpd_landmark_case.RData",sep=""))
rownames(pred) = colnames(lincs_signatures[, (split+1):ncol(lincs_signatures)])
colnames(pred) = rownames(lincs_signatures)
pred = t(pred)

load(paste(data_root, "/", "y_lincs_label_case.RData",sep=""))

cors_before = cor(lincs_signatures[, 1:split], lincs_signatures[, (split+1):ncol(lincs_signatures)])

cors_after = cor(lincs_signatures[, (split+1):ncol(lincs_signatures)], pred)

print(paste("before cor", mean(cors_before)))
print(paste("after cor", mean(cors_after)))

#tsne analysis
library(Rtsne) # Load package
samples = cbind(lincs_signatures[, 1:split], pred)
tsne_out = Rtsne(t(samples), perplexity = 10) # Run TSNE
cols = y
pdf(paste(model_root, "/", "pred/", model_id, "tsne.pdf", sep=""))
plot(tsne_out$Y, col = cols, 
     pch = c(rep(20, split), rep(8, ncol(pred)))) # Plot the result ,col=cell_tumor, pch=rges_bin c(rep(1, ncol(bad_signatures_subset)), rep(3, ncol(good_signatures_subset) ))
dev.off()

y_freq = table(y)
y_freq = names(y_freq > 1)
cors = cor(samples, samples)

cor_within_class = NULL
for (y_class in y_freq){
  cors_subset = cors[y == y_class, y == y_class]
  cor_within_class = c(cor_within_class, mean(cors_subset))
}
print(paste("ave cor within class", mean(cor_within_class)))

########################
###case study
lincs_sig_info_all <- read.csv(paste(data_root, "/lincs_sig_info_all.csv",sep=""), stringsAsFactors = F)
lincs_sig_info_all$pert_dose_bin = 10
lincs_sig_info_all$pert_dose_bin[lincs_sig_info_all$pert_dose < 9.5] = 0
lincs_sig_info_all$label = paste(lincs_sig_info_all$pert_iname, lincs_sig_info_all$cell_id, lincs_sig_info_all$pert_dose_bin, lincs_sig_info_all$pert_time, sep="_")
lincs_sig_info_all = lincs_sig_info_all[!duplicated(lincs_sig_info_all$id), ]
rownames(lincs_sig_info_all) = lincs_sig_info_all$id

#sample geldanamycin vorinostat
lincs_sig_info_subset = lincs_sig_info_all[lincs_sig_info_all$pert_iname %in% c("vorinostat") & lincs_sig_info_all$pert_dose == 10, ]

tsne_out = Rtsne(t(lincs_signatures[, colnames(lincs_signatures) %in% lincs_sig_info_subset$id]), perplexity = 30) # Run TSNE
cols = y[colnames(lincs_signatures) %in% lincs_sig_info_subset$id]
pdf(paste(model_root, "/", "pred/", model_id, "_vorinostat_tsne.pdf", sep=""))
plot(tsne_out$Y, col = cols, 
     pch = c(rep(20, sum(lincs_sig_info_subset$is_gold == 1)), rep(8, sum(lincs_sig_info_subset$is_gold == 0)))) # Plot the result ,col=cell_tumor, pch=rges_bin c(rep(1, ncol(bad_signatures_subset)), rep(3, ncol(good_signatures_subset) ))
dev.off()


