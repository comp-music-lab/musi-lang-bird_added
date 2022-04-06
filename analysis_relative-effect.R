###### library ######
library(ggplot2)
library(gridExtra)

###### setting ######
G_VIOLIN_SCALE <- "area"
G_VIOLIN_ADJUST <- 0.6
G_JITTER_WID <- 0.2
G_JITTER_ALP <- 0.4
G_JITTER_SIZE <- 3
G_WID <- 8
G_HEI <- 6

###### read data ######
dataname <- c('Ireland old style', 'Yangguan Sandie', 'Happy Birthday',
              'English_short', 'Sometimes behave so strangely', 'Vietnamese',
              'CANYO', 'FIREB', 'KAUAI')
labelname <- c('H2', 'H1', 'H3', 'S3', 'S1', 'S2', 'B3', 'B2', 'B1')
datatype = c('Music', 'Music', 'Music',
             'Speech', 'Speech', 'Speech',
             'Birdsong', 'Birdsong', 'Birdsong')
datadir <- './output/'
outputdir <- './output/'

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

###### plot ######
g <- ggplot(data = entropy_onsetwise, aes(x = Name, y = Entropy))
g <- g + geom_violin(trim = TRUE, scale = G_VIOLIN_SCALE, adjust = G_VIOLIN_ADJUST)
g <- g + geom_jitter(aes(color = Type), width = G_JITTER_WID, alpha = G_JITTER_ALP, size = G_JITTER_SIZE)
g <- g + scale_x_discrete(limits = data_ordered, labels = label_ordered)
g <- g + xlab('') + ggtitle('Onset-wise F0 entropy') + theme(plot.title = element_text(hjust = 0.5))

plot(g)

###### Output ######
ggsave(file = paste(outputdir, "figure_distribution.png", sep = ""), plot = g,
       width = G_WID, height = G_HEI)

###### plot ######
g <- ggplot(data = entropy_mean, aes(x = Entropy, y = Rating))
g <- g + geom_smooth(method = 'lm', formula = y~x)
g <- g + geom_point(aes(color = Type))
g <- g + ggtitle(paste('Mean entropy and human rating\n(Pearson\'s r = ', round(r, 2), ', slope = ', round(linearMod$coefficients[2], 2), ')', sep = '')) +
  theme(plot.title = element_text(hjust = 0.5))

plot(g)

###### Output ######
ggsave(file = paste(outputdir, "figure_correlation.png", sep = ""), plot = g,
       width = G_WID, height = G_HEI)