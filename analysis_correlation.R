###### library ######
library(ggplot2)
library(ggpubr)

###### setting ######
G_VIOLIN_SCALE <- "area"
G_VIOLIN_ADJUST <- 0.6
G_JITTER_WID <- 0.2
G_JITTER_ALP <- 0.4
G_JITTER_SIZE <- 3
G_XTICK_TEXT_SIZE <- 12
G_YTICK_TEXT_SIZE <- 12
G_XTITLE_TEXT_SIZE <- 13
G_YTITLE_TEXT_SIZE <- 13
G_LEGEND_TEXT_SIZE <- 12
G_WID <- 8
G_HEI <- 4
COLORPALETTE <- c("Human music" = "#56B4E9", "Human speech" = "#FF6600", "Birdsong" = "#009E73")

datadir <- './output/'
outputdir <- './output/'

dataname <- c('Ireland old style', 'Yangguan Sandie', 'Happy Birthday',
              'English_short', 'Sometimes behave so strangely', 'Vietnamese',
              'CANYO', 'FIREB', 'KAUAI')
labelname <- c('M2', 'M1', 'M3', 'S3', 'S1', 'S2', 'B3', 'B2', 'B1')
datatype <- c('Human music', 'Human music', 'Human music',
              'Human speech', 'Human speech', 'Human speech',
              'Birdsong', 'Birdsong', 'Birdsong')

###### read mean percentage error result ######
mean_percentage_error <- read.csv(paste(datadir, 'MeanPercentageError_results.csv', sep = ''))

mean_percentage_error$name[mean_percentage_error$name == 'Ireland Old Style'] <- dataname[1]
mean_percentage_error$name[mean_percentage_error$name == 'American English'] <- dataname[4]
mean_percentage_error$name[mean_percentage_error$name == '\'Sometimes behave so strangely\''] <- dataname[5]
mean_percentage_error$name[mean_percentage_error$name == 'Canyon wren'] <- dataname[7]
mean_percentage_error$name[mean_percentage_error$name == 'Firecrest'] <- dataname[8]
mean_percentage_error$name[mean_percentage_error$name == 'Kauai O\' o'] <- dataname[9]

if (!setequal(mean_percentage_error$name, dataname)) {
  print('Error: Data name is inconsistent')
  return()
}

###### read entropy result ######
entropy_ioiwise <- c()
entropy_mean <- c()

for (i in 1:length(dataname)) {
  filepath <- paste(datadir, dataname[i], '_H.csv', sep = "")
  tmp <- read.csv(filepath)
  tmp$Name <- dataname[i]
  tmp$Type <- datatype[i]
  entropy_ioiwise <- rbind(entropy_ioiwise, tmp)
  entropy_mean <- rbind(entropy_mean, data.frame(Entropy = weighted.mean(tmp$Entropy, tmp$Duration), Name = dataname[i], Type = datatype[i]))
}

entropy_ioiwise$Type <- factor(entropy_ioiwise$Type, levels = c("Human music", "Human speech", "Birdsong"))
entropy_mean$Type <- factor(entropy_mean$Type, levels = c("Human music", "Human speech", "Birdsong"))

###### read human rating result ######
ratinginfo <- read.csv('./data/PitchDiscretenessRating.csv')

###### combine data ######
pitchdiscreteness <- data.frame(Name = character(), Type = character(), MPE = numeric(), Entropy = numeric(), Rating = numeric())

for (i in 1:length(dataname)) {
  pitchdiscreteness[nrow(pitchdiscreteness) + 1, ] <- 
    list(dataname[i],
         datatype[i],
         mean_percentage_error$mean_percentage_error[mean_percentage_error$name == dataname[i]],
         entropy_mean$Entropy[entropy_mean$Name == dataname[i]],
         ratinginfo$Rating[ratinginfo$Audio == dataname[i]]
         )
}

pitchdiscreteness$Type <- factor(pitchdiscreteness$Type, levels = c("Human music", "Human speech", "Birdsong"))

###### plot - main result ######
idx <- sort(pitchdiscreteness$Rating, decreasing = TRUE, index = TRUE)$ix
data_ordered <- pitchdiscreteness$Name[idx]
label_ordered <- rep(c(''), length(data_ordered))
for (i in 1:length(data_ordered)) {
  label_ordered[i] <- labelname[data_ordered[i] == dataname]
}

g1 <- ggplot(data = pitchdiscreteness, aes(x = Name, y = Rating, color = Type))
g1 <- g1 + geom_line(aes(group = 1), color = 'gray') + geom_point() +
  theme(axis.title.x = element_blank()) + ylab("Human rating") + 
  scale_x_discrete(limits = data_ordered, labels = label_ordered) + 
  scale_y_continuous(breaks = c(4, 5, 6)) + theme(legend.title = element_blank())

g2 <- ggplot(data = pitchdiscreteness, aes(x = Name, y = MPE, color = Type))
g2 <- g2 + geom_line(aes(group = 1), color = 'gray') + geom_point() +
  theme(axis.title.x = element_blank()) + ylab("Mean percentage error") + 
  scale_x_discrete(limits = data_ordered, labels = label_ordered) + 
  scale_y_continuous(breaks = c(2, 4, 6)) + theme(legend.title = element_blank())

g3 <- ggplot(data = pitchdiscreteness, aes(x = Name, y = Entropy, color = Type))
g3 <- g3 + geom_line(aes(group = 1), color = 'gray') + geom_point() +
  theme(axis.title.x = element_blank()) + ylab("Weighted average entropy") + 
  scale_x_discrete(limits = data_ordered, labels = label_ordered) + 
  scale_y_continuous(breaks = c(3, 4, 5)) + theme(legend.title = element_blank())

g <- ggarrange(g1, g2, g3, font.label = list(size = 14, face = "plain", color ="black"),
               ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom")
g <- annotate_figure(g, top = text_grob("Pitch discreteness",  face = "bold", size = 16))
plot(g)

ggsave(file = paste(outputdir, "figure_pitchdiscreteness.png", sep = ""), plot = g, width = 8, height = 3)

###### plot - correlation ######
t1 <- cor.test(pitchdiscreteness$Rating, pitchdiscreteness$MPE, alternative = c("less"), method = "pearson", conf.level = 0.95)
t2 <- cor.test(pitchdiscreteness$Rating, pitchdiscreteness$Entropy, alternative = c("less"), method = "pearson", conf.level = 0.95)
t3 <- cor.test(pitchdiscreteness$MPE, pitchdiscreteness$Entropy, alternative = c("greater"), method = "pearson", conf.level = 0.95)

g1 <- ggplot(data = pitchdiscreteness, aes(x = Rating, y = MPE, color = Type))
g1 <- g1 + geom_point() + theme(legend.title = element_blank()) + 
  xlab("Human rating") + ylab("Mean percentage error") + 
  annotate(geom = "text", x = -Inf, y = Inf, hjust = -1.5, vjust = 1, label = sprintf('r = %3.2f', t1$estimate)) + 
  annotate(geom = "text", x = 4.66, y = 5.88, hjust = 0, vjust = 0, label = sprintf('p = %1.2f%%', t1$p.value*100))

g2 <- ggplot(data = pitchdiscreteness, aes(x = Rating, y = Entropy, color = Type))
g2 <- g2 + geom_point() + theme(legend.title = element_blank()) + 
  xlab("Human rating") + ylab("Weighted average entropy") + 
  annotate(geom = "text", x = -Inf, y = Inf, hjust = -1.5, vjust = 1, label = sprintf('r = %3.2f', t2$estimate)) + 
  annotate(geom = "text", x = 4.66, y = 5.27, hjust = 0, vjust = 0, label = sprintf('p = %1.2f%%', t2$p.value*100))

g3 <- ggplot(data = pitchdiscreteness, aes(x = MPE, y = Entropy, color = Type))
g3 <- g3 + geom_point() + theme(legend.title = element_blank()) + 
  xlab("Mean percentage error") + ylab("Weighted average entropy") + 
  annotate(geom = "text", x = -Inf, y = Inf, hjust = -1.5, vjust = 1, label = sprintf('r = %3.2f', t3$estimate)) + 
  annotate(geom = "text", x = 2.59, y = 5.27, hjust = 0, vjust = 0, label = sprintf('p = %1.2f%%', t3$p.value*100))

g <- ggarrange(g1, g2, g3, font.label = list(size = 14, face = "plain", color ="black"),
               ncol = 3, nrow = 1, common.legend = TRUE, legend = "bottom")
g <- annotate_figure(g, top = text_grob("Correlations",  face = "bold", size = 16))
plot(g)

ggsave(file = paste(outputdir, "figure_corr.png", sep = ""), plot = g, width = 8, height = 3)