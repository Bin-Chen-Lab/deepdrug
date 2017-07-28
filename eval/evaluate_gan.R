#evaluate a specific model, assess the correlation between sRGES and drug efficacy. The higher indicates the sample is better.

################
#para
args <- commandArgs(trailingOnly=T)

data_root <- args[1] #data folder
model_root <- args[2] #model folder
model_id <- args[3]
##################

#setwd("~/proj/deepdrug/data/example_data/")
library("plyr")
library("ggplot2")

################
#functions
cmap_score_new <- function(sig_up, sig_down, drug_signature) {
  #the old function does not support the input list with either all up genes or all down genes, this new function attempts to addess this.
  #we also modify the original CMap approach: whenever the sign of ks_up/ks_down, we substract the two scores such that the final scores would not enrich at 0.
  
  num_genes <- nrow(drug_signature)
  ks_up <- 0
  ks_down <- 0
  connectivity_score <- 0
  
  # I think we are re-ranking because the GeneID mapping changed the original rank range
  drug_signature[,"rank"] <- rank(drug_signature[,"rank"])
  
  # Merge the drug signature with the disease signature by GeneID. This becomes the V(j) from the algorithm description
  up_tags_rank <- merge(drug_signature, sig_up, by.x = "ids", by.y = 1)
  down_tags_rank <- merge(drug_signature, sig_down, by.x = "ids", by.y = 1)
  
  up_tags_position <- sort(up_tags_rank$rank)
  down_tags_position <- sort(down_tags_rank$rank)
  
  num_tags_up <- length(up_tags_position)
  num_tags_down <- length(down_tags_position)
  
  # 
  if(num_tags_up > 1) {
    a_up <- 0
    b_up <- 0
    
    a_up <- max(sapply(1:num_tags_up,function(j) {
      j/num_tags_up - up_tags_position[j]/num_genes
    }))
    b_up <- max(sapply(1:num_tags_up,function(j) {
      up_tags_position[j]/num_genes - (j-1)/num_tags_up
    }))
    
    if(a_up > b_up) {
      ks_up <- a_up
    } else {
      ks_up <- -b_up
    }
  }else{
    ks_up <- 0
  }
  
  if (num_tags_down > 1){
    
    a_down <- 0
    b_down <- 0
    
    a_down <- max(sapply(1:num_tags_down,function(j) {
      j/num_tags_down - down_tags_position[j]/num_genes
    }))
    b_down <- max(sapply(1:num_tags_down,function(j) {
      down_tags_position[j]/num_genes - (j-1)/num_tags_down
    }))
    
    if(a_down > b_down) {
      ks_down <- a_down
    } else {
      ks_down <- -b_down
    }
  }else{
    ks_down <- 0
  }
  
  if (ks_up == 0 & ks_down != 0){ #only down gene inputed
    connectivity_score <- -ks_down 
  }else if (ks_up !=0 & ks_down == 0){ #only up gene inputed
    connectivity_score <- ks_up
  }else if (sum(sign(c(ks_down,ks_up))) == 0) {
    connectivity_score <- ks_up - ks_down # different signs
  }else{
    connectivity_score <- ks_up - ks_down
  }
  
  return(connectivity_score)
}

getsRGES2 <- function(RGES, cor, pert_dose, pert_time){
  sRGES <- RGES 
  
  #older version
  if (pert_time < 24){
    sRGES <- sRGES - 0.1
  }
  
  if (pert_dose < 10){
    sRGES <- sRGES - 0.2
  }
  return(sRGES * cor)
}



cmd = paste("python ../denoise_gan_pred.py --cuda  --data_root", data_root, "--model_root", model_root, "--model_id", model_id, sep=" ")
print(cmd)
system(cmd)

pre_path = paste(model_root, "/", "pred/", model_id, ".txt", sep="")

while(!file.exists(pre_path)){
  Sys.sleep(30)
}

pred = read.csv(paste( model_root, "/", "pred/", model_id, ".txt", sep=""), sep="\t", header =F)
print(dim(pred))
#

#assign col/row names
load(paste(data_root, "/", "lincs_signatures_cmpd_landmark_all.RData",sep=""))
rownames(pred) = colnames(lincs_signatures[, 66512:ncol(lincs_signatures)])
colnames(pred) = rownames(lincs_signatures)
lincs_signatures = t(pred)

landmark <- 1
lincs_sig_info <- read.csv("lincs_sig_info_subset.csv")
sig.ids <- lincs_sig_info$id

##############
#read disease signatures
dz_signature <- read.csv("LIHC_sig_reduced.csv")
dz_genes_up <- subset(dz_signature,up_down=="up",select="GeneID")
dz_genes_down <- subset(dz_signature,up_down=="down",select="GeneID")
###############

##############
#compute RGES
#only support landmark genes
gene.list <- rownames(lincs_signatures)

#compute RGES
#only choose the top 150 genes
max_gene_size <- 150
if (nrow(dz_genes_up) > max_gene_size){
  dz_genes_up <- data.frame(GeneID= dz_genes_up[1:max_gene_size,])
}
if (nrow(dz_genes_down) > max_gene_size){
  dz_genes_down <- data.frame(GeneID=dz_genes_down[1:max_gene_size,])
}

dz_cmap_scores <- NULL
count <- 0
for (exp_id in sig.ids) {
  count <- count + 1
#  print(count)
  if (landmark ==1){
    cmap_exp_signature <- data.frame(gene.list,  rank(-1 * lincs_signatures[, as.character(exp_id)], ties.method="random"))    
  }else{
    cmap_exp_signature <- cbind(gene.list,  get.sigs(exp_id))    
  }
  colnames(cmap_exp_signature) <- c("ids","rank")
  dz_cmap_scores <- c(dz_cmap_scores, cmap_score_new(dz_genes_up,dz_genes_down,cmap_exp_signature))
}

pred <- data.frame(id = sig.ids, cmap_score = dz_cmap_scores)

pred <- merge(pred, lincs_sig_info, by = "id")

pred$cor <- 1
pred$RGES <- sapply(1:nrow(pred), function(id){getsRGES2(pred$cmap_score[id], pred$cor[id], pred$pert_dose[id], pred$pert_time[id])})

pred_merged <- ddply(pred,  .(pert_iname),  summarise,
                     mean = mean(RGES),
                     n = length(RGES),
                     median = median(RGES),
                     sd = sd(RGES))
pred_merged$sRGES <- pred_merged$mean

#write.csv(pred_merged, sRGES_output_path)

##############
#sRGES and drug efficacy

RGES_efficacy <- merge(pred_merged, unique(lincs_sig_info[, c("pert_iname", "standard_value")]), by = "pert_iname")

cor_test <- cor.test(RGES_efficacy$sRGES, log(RGES_efficacy$standard_value, 10), method="spearman", exact=FALSE) #
lm_cmap_ic50 <- lm( log(standard_value, 10) ~ sRGES, RGES_efficacy)

print(paste("correlation between RGES and efficacy is ", cor_test$estimate))

pdf_path =  paste(model_root, "/", "pred/", model_id, "_cor.pdf", sep="")
pdf(pdf_path)
g <- ggplot(RGES_efficacy, aes(sRGES, log(RGES_efficacy$standard_value, 10)  )) +  theme_bw()  + 
  theme(legend.position ="bottom", axis.text=element_text(size=18), axis.title=element_text(size=18)) +                                                                                               
  stat_smooth(method="lm", se=F, color="black")  + geom_point(size=3) + 
  annotate("text", label = paste("r=", format(summary(lm_cmap_ic50)$r.squared ^ 0.5, digit=2), ", ",  "P=", format(anova(lm_cmap_ic50)$`Pr(>F)`[1], digit=3, scientific=T), sep=""), x = 0, y = 7.5, size = 6, colour = "black") +
  annotate("text", label = paste("rho=", format(cor_test$estimate, digit=2), ", P=", format(cor_test$p.value, digit=3, scientific=T), sep=""), x = 0, y = 7.1, size = 6, colour = "black") +
  scale_size(range = c(2, 5)) +
  xlab("sRGES") + guides(shape=FALSE, size=FALSE) +
  ylab("log10(IC50) nm") + coord_cartesian(xlim = c(-0.7, 0.7), ylim=c(-1, 8))
print(g)
dev.off()

