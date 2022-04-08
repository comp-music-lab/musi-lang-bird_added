###### library ######
library(ggplot2)
library(ggpubr)

###### setting ######
G_VIOLIN_SCALE <- "area"
G_VIOLIN_ADJUST <- 0.6
G_JITTER_WID <- 0.2
G_JITTER_ALP <- 0.4
G_JITTER_SIZE <- 3
G_XTICK_TEXT_SIZE <- 14
G_YTICK_TEXT_SIZE <- 14
G_XTITLE_TEXT_SIZE <- 16
G_YTITLE_TEXT_SIZE <- 16
G_LEGEND_TEXT_SIZE <- 15
G_WID <- 8
G_HEI <- 4

###### read data ######
dataname <- c('Ireland old style', 'Yangguan Sandie', 'Happy Birthday',
              'English_short', 'Sometimes behave so strangely', 'Vietnamese',
              'CANYO', 'FIREB', 'KAUAI')
labelname <- c('H2', 'H1', 'H3', 'S3', 'S1', 'S2', 'B3', 'B2', 'B1')
datatype <- c('Human music', 'Human music', 'Human music',
             'Human speech', 'Human speech', 'Human speech',
             'Birdsong', 'Birdsong', 'Birdsong')
datadir <- './output/'
outputdir <- './output/'

datatype <- as.factor(datatype)
levels(datatype) <- c("Human speech", "Birdsong", "Human music")

entropy_onsetwise <- c()
entropy_mean <- c()

for (i in 1:length(dataname)) {
  filepath <- paste(datadir, dataname[i], '_H.csv', sep = "")
  tmp <- read.csv(filepath)
  tmp$Name <- dataname[i]
  tmp$Type <- datatype[i]
  entropy_onsetwise <- rbind(entropy_onsetwise, tmp)
  entropy_mean <- rbind(entropy_mean, data.frame(Entropy = mean(tmp$Entropy), Name = dataname[i], Type = datatype[i]))
}
entropy_mean$Rating <- 0

ratinginfo <- read.csv('./data/PitchDiscretenessRating.csv')

idx <- sort(ratinginfo$Rating, decreasing = TRUE, index = TRUE)$ix
data_ordered <- ratinginfo$Audio[idx]
label_ordered <- rep(c(''), length(data_ordered))
for (i in 1:length(data_ordered)) {
  label_ordered[i] <- labelname[data_ordered[i] == dataname]
}

###### correlation ######
for (i in 1:length(dataname)) {
  entropy_mean[ratinginfo$Audio[i] == entropy_mean$Name, 4] <- ratinginfo$Rating[i]
}

r <- cor(entropy_mean$Entropy, entropy_mean$Rating, method = "pearson")
linearMod <- lm(Rating ~ Entropy, data = entropy_mean) 

print(paste('Pearson\'s r = ', round(r, 3), ', slope = ', round(linearMod$coefficients[2], 3), sep = ''))

###### plot ######
g1 <- ggplot(data = entropy_onsetwise, aes(x = Name, y = Entropy))
g1 <- g1 + geom_violin(trim = TRUE, scale = G_VIOLIN_SCALE, adjust = G_VIOLIN_ADJUST)
g1 <- g1 + geom_jitter(aes(color = Type), width = G_JITTER_WID, alpha = G_JITTER_ALP, size = G_JITTER_SIZE)
g1 <- g1 + scale_x_discrete(limits = data_ordered, labels = label_ordered)
g1 <- g1 + xlab('') + ggtitle('IOI-wise F0 entropy') + theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(size = G_XTICK_TEXT_SIZE), axis.text.y = element_text(size = G_YTICK_TEXT_SIZE),
        axis.title.x = element_text(size = G_XTITLE_TEXT_SIZE), axis.title.y = element_text(size = G_YTITLE_TEXT_SIZE)) + 
  theme(legend.title = element_blank(), legend.text = element_text(size = G_LEGEND_TEXT_SIZE), legend.position = "bottom")

plot(g1)

###### Output ######
ggsave(file = paste(outputdir, "figure_distribution.png", sep = ""), plot = g1, width = G_WID, height = G_HEI)

###### plot ######
g2 <- ggplot(data = entropy_mean, aes(x = Entropy, y = Rating))
g2 <- g2 + geom_smooth(method = 'lm', formula = y~x)
g2 <- g2 + geom_point(aes(color = Type))
g2 <- g2 + ggtitle('Mean entropy and human rating') +
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(size = G_XTICK_TEXT_SIZE), axis.text.y = element_text(size = G_YTICK_TEXT_SIZE),
        axis.title.x = element_text(size = G_XTITLE_TEXT_SIZE), axis.title.y = element_text(size = G_YTITLE_TEXT_SIZE)) + 
  theme(legend.position = "none")

plot(g2)

###### Output ######
ggsave(file = paste(outputdir, "figure_correlation.png", sep = ""), plot = g2, width = G_WID, height = G_HEI)

###### Combining ######
g <- ggarrange(g1, g2,  labels = c("D", "E"), ncol = 2, nrow = 1, common.legend = TRUE)

###### Output ######
ggsave(file = paste(outputdir, "figure_combined.png", sep = ""), plot = g, width = G_WID, height = G_HEI)