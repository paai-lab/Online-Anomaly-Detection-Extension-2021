# install.packages('keras')
# install.packages('purrr')
# install.packages('functional')

library(MASS)
library(caret)
library(fGarch)
library(fitdistrplus)
library(pracma)
library(BBmisc)
library(functional)
library(dplyr)
library(keras)
library(lubridate)
library(tensorflow)
Sys.sleep(5)
install_tensorflow(restart_session = FALSE)


setwd("/home/jonghyeon3/extension_AD/evaluations/data")
fn<-list.files(getwd())

fn = fn[c(1,3,7,8,9,10,12,13,15,16)]
fn
index= 1:10
result = as.data.frame(index  )

#functions
{
  fun_leverage = function(x){
    A<- ginv(t(x)%*%x)
    H_part1<- x%*%A  
    h_diag <- colSums(t(H_part1)*t(x))
    return(h_diag)
  }
  
  fun_embedding = function(ActivityID, embedding_size){
    model <- keras_model_sequential()
    model %>% layer_embedding(input_dim = length(unique(ActivityID))+1, output_dim = embedding_size, input_length = 1, name="embedding") %>%
      layer_flatten()  %>%  
      layer_dense(units=40, activation = "relu") %>%  
      layer_dense(units=10, activation = "relu") %>%  
      layer_dense(units=1)
    model %>% compile(loss = "mse", optimizer = "sgd", metric="accuracy")
    layer <- get_layer(model, "embedding")
    embeddings <- data.frame(layer$get_weights()[[1]])
    embeddings$ActivityID <- c("none", levels(ActivityID) )
    return(embeddings)
  }
  
  fun_onehot = function(data){
    if(length(levels(data$ActivityID))>1){
      a<- model.matrix(~ActivityID, data = data)
      A<- as.numeric(data[,2])
      A[which(A!=1)] <- 0
      a<- cbind(ActivityID1 = A, a[,-1])
      onehot<- as.data.frame(a)
    }else{
      A<- as.numeric(data[,2])
      A[which(A!=1)] <- 0
      a<- cbind(ActivityID1 = A)
      onehot<- as.data.frame(a)
    }
    return(onehot)
  }
}


  #data load and preprocess
for(k in 1:10){
    
    input = data.frame(read.csv(fn[k], header=T))
    normal= input[which(input$anomaly_type =="normal"),]
    anomaly= input[which(input$anomaly_type !="normal"),]
    normal_seq = aggregate(normal$Activity, by=list(normal$Case), FUN=paste0)
    anomaly_seq = aggregate(anomaly$Activity, by=list(anomaly$Case), FUN=paste0)
    delete_case= anomaly_seq[which(is.element(anomaly_seq$x , normal_seq$x)),'Group.1']
    input = input[which(!is.element(input$Case, delete_case)),]
    input$Event = 1:nrow(input)
    input$Event = as.factor(input$Event)
    one= rep(1, nrow(input))
    input[,'start'] = ave(one, by= input$Case, FUN= cumsum) -1
    input[which(input$start !=1),'start'] =0
  
  ####
  
  
    pre<-input
    pre= pre[ with(pre, order(Case,timestamp)),]
    one= rep(1, nrow(pre))
    pre[,'start'] = ave(one, by= pre$Case, FUN= cumsum) -1
    pre[which(pre$start !=1),'start'] =0
    pre= pre[ with(pre, order(timestamp)),]
    pre[,'Event'] = as.factor(1:nrow(pre))
    pre[,'num_case'] = cumsum(pre$start)
    pre[,'leverage'] = rep(-1, nrow(pre))
    pre[,'TF'] = rep(0, nrow(pre))
    pre[,'w-TF'] = rep(0, nrow(pre))
    pre[,'t3'] = rep(0, nrow(pre))
    pre[,'tn']= rep(0, nrow(pre))
    pre[,'time'] = rep(0, nrow(pre))
    event_num = nrow(pre)
    case_num= length(unique(pre$Case))
    last_index = nrow(pre)

    leverage_start <- Sys.time()
    pre2 = pre[which(!is.na(pre$Case)),]
    
    subresult= as.numeric()
    for(i in 1:max(pre2$order)){
      candi=  pre2[which(pre2$order == i), "Case"]
      pre2= pre2[is.element(pre2$Case, candi),]
      data <- pre2[which(pre2$order <= i),c("Case","Activity","order", "Event")] 
      c=unique(pre2[which(pre2$order <= i),c("Case","anomaly_type")])
      names(data)[1:2] <- c("ID", "ActivityID") 
      
      data$ID <- as.factor(data$ID)
      data$ActivityID <- as.factor(data$ActivityID)
      
      # One-hot encoding
      data1 <- fun_onehot(data)      
      newdat <- cbind(data[,1], data1)
      newdat[,1] <- as.factor(newdat[,1])
      n<- length(levels((newdat[,1])))   # the number of cases
      m<-max(table((newdat[,1])))     # maximum trace length
      num_act= ncol(newdat)-1
      num_event = nrow(newdat)
      max<- m*num_act
      newdat2<- matrix(NA, nrow=n , ncol=max)
      for(j in 1:n){
        cut = newdat[which(newdat[,1]== c[j,1] ),-1]
        save2 <- as.vector(t(cut))
        newdat2[j,1:length(save2)] <- save2
      }
      
      newdat3= cbind(c, newdat2)
      x2= newdat3[,-c(1:2)]
      x= as.matrix(sapply(x2, as.numeric))  
      
      cc= cbind(c, act= (c$anomaly_type != "normal") , pred = rep(0, nrow(c)))
  
      #Calculate leverage
      if(nrow(x) ==1){
        h_diag = 0
      }else{
        h_diag <- fun_leverage(x)
      }
      
      cc= cbind(c, act= (c$anomaly_type != "normal") ,pred = (h_diag>(mean(h_diag)+sd(h_diag))))
      

      subresult= rbind(subresult, cc)
    }

    
    cm4 <- confusionMatrix( as.factor(subresult$pred), as.factor(subresult$act),positive = 'TRUE')
    cm4 <- as.vector(cm4[4])[[1]]
    

    precision4  <- cm4[5]
    Recall4 <- cm4[6]
    Fs4 <- cm4[7]

    
    result$name[k] = fn[k]
    result$acc[k] =  sum( subresult$pred==subresult$act)/nrow(subresult)
    result$prec[k] = precision4
    result$recall[k] = Recall4
    result$fs[k] = Fs4
}

setwd("~/extension_AD/evaluations/offline")

write.csv(result, "offline_leverage.csv", row.names = F)

