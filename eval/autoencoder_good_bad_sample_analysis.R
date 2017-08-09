library(Rtsne) # Load package

args <- commandArgs(trailingOnly=T)

data_root <- args[1]

pre_path = paste( data_root, "/", "decoded_data.txt", sep="")

pred = read.csv(pre_path, sep="\t", header =F)
print(dim(pred))
#last 10K
#pred = pred[(nrow(pred)-9999):nrow(pred), ]
load(paste(data_root, "/", "lincs_signatures_cmpd_landmark_all.RData", sep=""))


cors = NULL
for (i in 1:nrow(pred)){ #
  #plot(as.numeric(decoded[i,]), as.numeric(lincs_signatures[,i]))
# print(i)
  cors = c(cors, cor(as.numeric(pred[i,]), as.numeric(lincs_signatures[,56512 + i])))
}

print(paste("mean between input and output samples for good samples", mean(cors[1:1000])))
print(paste("mean between input and output samples for bad samples", mean(cors[1001:length(cors)])))


