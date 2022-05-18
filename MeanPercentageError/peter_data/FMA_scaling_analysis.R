# ordering conditions for FMA scaling study

basedir <- '/Users/PQP/MyFiles2019/VocalScales/FMA_stimuli/'
infile <- 'FMA_scaling_data.csv'

datanow <- read.csv(paste(basedir, infile, sep = ""))
# create varaibles for stim categories
# a shrotened ID for stimuli
datanow$stim <- substr(as.character(datanow$file),1,3)
# category of bird, human music, human song
datanow$stimcat <- substr(as.character(datanow$file),1,2)
# original rank order
datanow$stimrank <- substr(as.character(datanow$file),3,3)

# try a boxplot to start
boxplot(datanow$rating ~ datanow$stim, range=0)
#
# number conditions for plotting
datanow$ncond <- 1
datanow$ncond[datanow$stim == 'bs2'] <- 2
datanow$ncond[datanow$stim == 'bs3'] <- 3
datanow$ncond[datanow$stim == 'hm1'] <- 4
datanow$ncond[datanow$stim == 'hm2'] <- 5
datanow$ncond[datanow$stim == 'hm3'] <- 6
datanow$ncond[datanow$stim == 'hs1'] <- 7
datanow$ncond[datanow$stim == 'hs2'] <- 8
datanow$ncond[datanow$stim == 'hs3'] <- 9
#
# points
points(rnorm(length(datanow$rating), datanow$ncond, 0.25),
      datanow$rating,
      pch = 20, col = "blue")

# some stats?
library(ez)
source('/Users/PQP/MyFiles2019/Rscripts/PrintANOVA_PQP.R')
PrintANOVA_PQP(ezANOVA(data=datanow, wid=participant, within =.(stimcat, stimrank), dv=rating, detailed = TRUE))
source('/Users/PQP/MyFiles2019/Rscripts/PrintStats_PQP.R')
toplot <- PrintStats_PQP(ezStats(data=datanow, wid=participant, within =.(stimcat, stimrank), dv=rating))
PrintStats_PQP(ezStats(data=datanow, wid=participant, within =.(stimcat), dv=rating))


# try a table
library(reshape)
tablenow <- cast(datanow[,c(1, 4, 5)], file ~ participant, FUN=mean)
tablenow$M <- rowMeans(tablenow)
print(tablenow)


#re-order to match John's plot
toplot$orderplot <- c(7,8,9,1,2,3,4,5,6)
toplot <- toplot[order(toplot$orderplot),]

#try creating plot to match John's
library(gplots)
barplot2(toplot$Mean,
         space = .5, 
         ylim = c(1,7), xpd = FALSE,
#         names.arg = c('B1', "B2", "B3", "M1", "M2", "M3", "S1", "S2", "S3"),
         names.arg = c( "M1", "M2", "M3", "S1", "S2", "S3", 'B1', "B2", "B3"),
         xlab = " ",
         ylab = "Rating of discreteness",
         axis.lty = 1,
         plot.ci = TRUE,
         ci.u = toplot$Mean + toplot$SE,
         ci.l = toplot$Mean - toplot$SE,
         cex.lab = 1.25, cex.main = 1.5, cex.axis = .75,
         col = c(rgb(0, 0, 1, 1), rgb(0, 0, 1, 1), rgb(0, 0, 1, 1),
                 rgb(1, 0, 0, 1), rgb(1, 0, 0, 1), rgb(1, 0, 0, 1),
                 rgb(0, 1, 0, 1), rgb(0, 1, 0, 1), rgb(0, 1, 0, 1))
         )

# stripchart(datanow$rating ~ datanow$stim, 
#            vertical = TRUE, method = "jitter", pch = 20)
