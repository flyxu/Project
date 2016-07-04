library(party)
library(BBmisc)
library(C50)
library(pROC)
library(ggplot2)
library(DMwR)
Data<-read.csv("./input/test.csv",header = TRUE)
Data$JGDZ2[Data$JGDZ=="北京"]<-1
Data$JGDZ2[Data$JGDZ=="上海"]<-2
Data$JGDZ2[Data$JGDZ=="浙江"]<-3
Data$JGDZ2[Data$JGDZ=="深证"]<-4
Data$JGDZ2[Data$JGDZ=="南京"]<-5
Data$JGDZ2[Data$JGDZ=="苏州"]<-6
Data$JGDZ2[Data$JGDZ=="无锡"]<-7
Data$JGDZ2[Data$JGDZ=="常州"]<-8
Data$JGDZ2[Data$JGDZ=="徐州"]<-9
Data$JGDZ2[Data$JGDZ=="扬州"]<-10
Data$JGDZ2[Data$JGDZ=="泰州"]<-11
Data$JGDZ2[Data$JGDZ=="南通"]<-12
Data$JGDZ2[Data$JGDZ=="镇江"]<-13
Data$JGDZ2[Data$JGDZ=="连云港"]<-14
Data$JGDZ2[Data$JGDZ=="淮安"]<-15
Data$JGDZ2[Data$JGDZ=="宿迁"]<-16
Data$JGDZ2[Data$JGDZ=="盐城"]<-17
Data$JGDZ2[Data$JGDZ=="无"]<-18

Data$ZYDBFS2[Data$ZYDBFS=="保证"]<-1
Data$ZYDBFS2[Data$ZYDBFS=="抵押"]<-2
Data$ZYDBFS2[Data$ZYDBFS=="信用"]<-3
Data$ZYDBFS2[Data$ZYDBFS=="质押（含保证金）"]<-4
Data$ZYDBFS2[Data$ZYDBFS=="其他"]<-5

Data$DKWJFL2[Data$DKWJFL=="正常"]<-"A"
Data$DKWJFL2[Data$DKWJFL=="不良"]<-"B"
newdata<-Data[c(-1:-2,-5:-8)]
newdata$DKWJFL2<-as.factor(newdata$DKWJFL2)
newdata$SSXY<-as.factor(newdata$SSXY)
newdata$SSXY<-as.numeric(newdata$SSXY)

newdata2<-normalize(newdata,method = "scale")
newdata2$DKWJFL2<-newdata$DKWJFL2

ind<-sample(2,nrow(newdata2),replace=TRUE,prob=c(0.7,0.3))
traindata<-newdata2[ind==1,]
testdata<-newdata2[ind==2,]

myFormula<-DKWJFL2~DKQX+SSXY+JE+JGDZ2+ZYDBFS2
smotedata<-SMOTE(myFormula,newdata2,perc.over=200,perc.under=150)
#查看数据的平衡状态
table(smotedata$DKWJFL2)
prop.table(table(smotedata$DKWJFL2))
set.seed(1234)
ind<-sample(2,nrow(newdata2),replace=TRUE,prob=c(0.7,0.3))
#traindata<-newdata2[ind==2,]
testdata<-newdata2[ind==1,]
i_tree<-C5.0(myFormula,data=smotedata)
testpred<-predict(i_tree,newdata=newdata2)
#构建模型评估的混淆矩阵
con<-table(newdata2$DKWJFL2,testpred)
con
#计算预测精度
Accuracy<-sum(diag(con))/sum(con)
Accuracy
#模型预测的ROC曲线
roc_curve<-roc(as.numeric(newdata2$DKWJFL2),as.numeric(testpred))
print(roc_curve)
x<-1-roc_curve$specificities
y<-roc_curve$sensitivities
p<-ggplot(data=NULL,mapping=aes(x=x,y=y))
p + geom_line(colour = 'red', size = 2) +geom_abline(intercept = 0, slope = 1)+ annotate('text', x = 0.4, y = 0.5, label=paste('AUC=',round(roc_curve$auc,2)), size = 6) + labs(x = '1-specificities',y = 'sensitivities', title = 'ROC Curve')

