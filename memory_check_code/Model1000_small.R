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
library(profmem)
Sys.sleep(5)
install_tensorflow(restart_session = FALSE)


setwd("/home/jonghyeon3/extension_AD/evaluations/data")
fn<-list.files(getwd())


#data load and preprocess
{
input = data.frame(read.csv("small-0.1-1.csv", header=T))
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
}
####


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
  
  fun_batch_remove_TRUE = function(input, Min, start_index, Max, until, embedding_size_p, remove_threshold ){}
  
  fun_batch_remove_FALSE = function(input, Min,start_index, Max, until, embedding_size_p ){
    #prepare data
    
    p = profmem({
      pre<-input
      pre= pre[ with(pre, order(Case,timestamp)),]
      one= rep(1, nrow(pre))
      pre[,'start'] = ave(one, by= pre$Case, FUN= cumsum) -1
      pre[which(pre$start !=1),'start'] =0
      pre= pre[ with(pre, order(timestamp)),]
      pre[,'Event'] = as.factor(1:nrow(pre))
      pre[,'num_case'] = cumsum(pre$start)
      pre[,'leverage'] = rep(-1, nrow(pre))
      pre[,'t1'] = rep(0, nrow(pre))
      pre[,'t2'] = rep(0, nrow(pre))
      pre[,'t3'] = rep(0, nrow(pre))
      pre[,'tn']= rep(0, nrow(pre))
      pre[,'time'] = rep(0, nrow(pre))
      pre[,'Mb'] = rep(-1, nrow(pre)); pre[,'Nc'] = rep(-1, nrow(pre)); pre[,'Ne'] = rep(-1, nrow(pre)); pre[,'MMb'] = rep(-1, nrow(pre))
      event_num = nrow(pre)
      case_num= length(unique(pre$Case))
      start_index  = start_index
      last_index = nrow(pre)
      
      leverage_start <- Sys.time()
      pre2 = pre[1:start_index,]
      cur_len = sum(pre2$start)
      data<- pre2[,c("Case","Activity","order")]  
      names(data)[1:2] <- c("ID", "ActivityID") 
      
      #basic: Max should be larger than Min or equal 
      if(Max< (Min+1)){
        Max=Min+1
      }
      
      # Max option
      if(cur_len > Max ){
        del_case = pre[which(pre$start==1),'Case'][1:(cur_len-Max)] 
        pre = pre[which(!is.element(pre$Case, del_case)),]
        pre[,'num_case'] = cumsum(pre$start)
        event_num = nrow(pre)
        case_num= length(unique(pre$Case))
        last_index = nrow(pre)
        pre2 = pre2[which(!is.element(pre2$Case, del_case)),]
        data<- pre2[,c("Case","Activity","order")]  
        names(data)[1:2] <- c("ID", "ActivityID") 
        cur_len = sum(pre2$start)
        start_index = nrow(pre2)
        last_index = nrow(pre)
      }
      
      
      if(embedding_size_p>0){
        num_act= length(unique(data$ActivityID))
        embedding_size = round(num_act*embedding_size_p)
        
        # deep embedding encoding
        embeddings = fun_embedding(as.factor(data$ActivityID), embedding_size)
        object_case = pre2$Case[nrow(pre2)]
        object_event = pre2$Event[nrow(pre2)]
        data$ID <- as.factor(data$ID)
        data$ActivityID <- as.factor(data$ActivityID)
        n= length(unique(data[,1]))
        m = max(table(data[,1]))
        data$order = as.character(data$order)
        data$ID = as.character(data$ID)
        
        all3 = merge(data, embeddings, by='ActivityID', all.x=T)
        all3= all3[ with(all3, order(ID, order)),]
        all3 = all3[,c("ID","ActivityID",names(all3)[(ncol(all3)-embedding_size+1):ncol(all3)])]
        
        num_event = nrow(all3)
        max<- m*(embedding_size)
        c=unique(pre2[,c("Case","anomaly_type")]) #CHANGE
        label = as.character(c[,2])
        
        # prefix encoding
        prefixL = as.numeric()
        newdat2<- matrix(NA, nrow=num_event , ncol=max)  
        for(j in 1:num_event){
          cut = all3[which(all3[1:j,1]== all3[j,1] ),-c(1:2)]
          if(class(cut)=='numeric'){
            prefixL[j] = 1
          }else{
            prefixL[j] = nrow(cut)
          }       
          save2 <- as.vector(t(cut))  
          newdat2[j,1:length(save2)] <- save2
        }
        newdat2[which(is.na(newdat2))] <- 0 # zero-padding
        newdat2_save= newdat2
        newdat3 = data.frame(cbind(Case=as.character(all3[,1]), label= as.character(pre2$anomaly_type), newdat2))
        
        x2= newdat3[which(prefixL == prefixL[start_index]),-(1:2)]
        x2 = x2[,1:(prefixL[start_index]*embedding_size)]
        
      }else{
        object_case = pre2$Case[nrow(pre2)]
        object_event = pre2$Event[nrow(pre2)]
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
        c=unique(pre2[,c("Case","anomaly_type")])
        
        # prefix encoding
        prefixL = as.numeric()
        newdat2<- matrix(NA, nrow=num_event , ncol=max)  
        for(j in 1:num_event){
          cut = newdat[which(newdat[1:j,1]== newdat[j,1] ),-1]
          if(class(cut)=='numeric'){
            prefixL[j] = 1
          }else{
            prefixL[j] = nrow(cut)
          }        
          save2 <- as.vector(t(cut))  
          newdat2[j,1:length(save2)] <- save2
        }
        newdat2[which(is.na(newdat2))] <- 0 # zero-padding
        newdat2_save= newdat2
        act_save = names(newdat) #change 1
        newdat3 = data.frame(cbind(Case=as.character(newdat[,1]), label= as.character(pre2$anomaly_type), newdat2))
        
        x2= newdat3[which(prefixL == prefixL[start_index]),-(1:2)]
        x2 = x2[,1:(prefixL[start_index]*num_act)]
      }
      
      #Caculate leverage
      x= as.matrix(sapply(x2, as.numeric))  
      h_diag <- fun_leverage(x)
    })
    if(sum(h_diag)!=0){pre[start_index, 'Mb'] = sum(p$bytes/10^6, na.rm=T)}else{pre[start_index, 'Mb'] = NA}
    if(sum(h_diag)!=0){pre[start_index, 'MMb'] = max(p$bytes/10^6, na.rm=T)}else{pre[start_index, 'MMb'] = NA}
    pre[start_index, 'Nc'] = sum(pre2$start)
    pre[start_index, 'Ne'] = num_event
    pre[start_index, 'leverage'] = h_diag[length(h_diag)]
    
    leverage_end <- Sys.time()
    
    pre[start_index, 'time'] =   (leverage_end-leverage_start)
    pre[start_index, 'tn'] = (h_diag[length(h_diag)] > (mean(h_diag)+sd(h_diag)))
    
    #Set escape option
    if(until==0 | start_index+until>last_index){
      until = last_index
    }else{
      until= start_index+until
    }
    
    #Start event steam 
    for(i in (start_index+1):until){ # last_index
      print(paste("Start to calculate leverage score of ", i ,"-th event (total ",event_num," events)" ,sep=''))
      leverage_start <- Sys.time()
      p = profmem({
        pre2 = rbind(pre2, pre[i,])
        cur_len = sum(pre2$start)
        data<- pre2[,c("Case","Activity",'order')]  
        names(data)[1:2] <- c("ID", "ActivityID") 
        
        # Max option
        object_case = pre2$Case[nrow(pre2)]
        object_event = pre2$Event[nrow(pre2)]
        if(cur_len > Max ){
          del_case = pre2[which(pre2$start==1),'Case']
          del_case = del_case[1:(cur_len-Max)] 
          del_case= del_case[which(!is.element(del_case, object_case))]
          data = data[which(!is.element(data[,1], del_case)),]
          pre3= pre2[which(!is.element(pre2[,1], del_case)),]
          label = as.character(pre3[,c("anomaly_type")])
          pre[i, 'Nc'] = sum(pre3$start)
        }else{
          label = as.character(pre2[,c("anomaly_type")])
          pre[i, 'Nc'] = sum(pre2$start)
        }
        
        if(embedding_size_p>0){
          num_act= length(unique(data$ActivityID))
          embedding_size = round(num_act*embedding_size_p)
          
          # embedding encoding
          embeddings = fun_embedding( as.factor(data$ActivityID), embedding_size)
          object_case = pre2$Case[nrow(pre2)]
          object_event = pre2$Event[nrow(pre2)]
          data$ID <- as.factor(data$ID)
          data$ActivityID <- as.factor(data$ActivityID)
          n= length(unique(data[,1]))
          m = max(table(data[,1]))
          data$order = as.character(data$order)
          data$ID = as.character(data$ID)
          
          all3 = merge(data, embeddings, by='ActivityID', all.x=T)
          all3= all3[ with(all3, order(ID, order)),]
          all3 = all3[,c("ID","ActivityID",names(all3)[(ncol(all3)-embedding_size+1):ncol(all3)])]
          
          num_event = nrow(all3)
          max<- m*(embedding_size)
          c=unique(pre2[,c("Case","anomaly_type")]) #CHANGE
          label = as.character(c[,2])
          
          { # update event
            newdat2<- matrix(NA, nrow=num_event , ncol=max)  
            prefixL = as.numeric()
            for(j in 1:num_event){
              cut = all3[which(all3[1:j,1]== all3[j,1] ),-c(1:2)]
              if(class(cut)=='numeric'){
                prefixL[j] = 1
              }else{
                prefixL[j] = nrow(cut)
              }           
              save2 <- as.vector(t(cut))  
              newdat2[j,1:length(save2)] <- save2
            }
          }
          
          # Max option
          if(cur_len > Max ){
            del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
            del_case= del_case[which(!is.element(del_case, object_case))]
            pre2 = pre2[which(!is.element(all3[,1], del_case)),]
            newdat2 = newdat2[which(!is.element(all3[,1], del_case)),]
            label= label[which(!is.element(all3[,1], del_case))]
            prefixL= prefixL[which(!is.element(all3[,1], del_case))]
            all3 = all3[which(!is.element(all3[,1], del_case)),]
          }   
          newdat2[which(is.na(newdat2))] <- 0 # zero-padding
          newdat2_save= newdat2
          newdat3 <-data.frame(cbind(Case= as.character(all3[,1]), label= label, newdat2))
          
          x2= newdat3[which(prefixL == prefixL[length(prefixL)]),-(1:2)]
          x2 = x2[,1:(prefixL[length(prefixL)]*embedding_size)]
          
        }else{
          object_case = pre2$Case[nrow(pre2)]
          object_event = pre2$Event[nrow(pre2)]
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
          newdat2<- matrix(NA, nrow=num_event , ncol=max)  
          prefixL = as.numeric()
          for(j in 1:num_event){
            cut = newdat[which(newdat[1:j,1]== newdat[j,1] ),-1]
            if(class(cut)=='numeric'){
              prefixL[j] = 1
            }else{
              prefixL[j] = nrow(cut)
            }           
            save2 <- as.vector(t(cut))  
            newdat2[j,1:length(save2)] <- save2
          }
          
          act_save = names(newdat) #change 1
          newdat2[which(is.na(newdat2))] <- 0 # zero-padding
          newdat2_save= newdat2
          newdat3 <-data.frame(cbind(Case= as.character(newdat[,1]), label= label, newdat2))
          
          x2= newdat3[which(prefixL == prefixL[length(prefixL)]),-(1:2)]
          x2 = x2[,1:(prefixL[length(prefixL)]*num_act)]
        }
        
        #Calculate leverage
        x= as.matrix(sapply(x2, as.numeric))  
        h_diag <- fun_leverage(x)
        pre[i, 'leverage'] = h_diag[length(h_diag)]
        leverage_end <- Sys.time()
        print(paste("Anomaly score of", i ,"-th event = ", round( h_diag[length(h_diag)],5), " (CaseID=",object_case,")" ,sep=''))
        pre[i, 'time'] =   (leverage_end-leverage_start)
        pre[i, 'tn'] = (h_diag[length(h_diag)] > (mean(h_diag)+sd(h_diag)))
      })
      if(sum(h_diag)!=0){pre[i, 'Mb'] = sum(p$bytes/10^6, na.rm=T)}else{pre[i, 'Mb'] = NA}
      if(sum(h_diag)!=0){pre[i, 'MMb'] = max(p$bytes/10^6, na.rm=T)}else{pre[i, 'MMb'] = NA}
      pre[i, 'Ne'] = num_event
    }
    return(pre)
    
  }
  
  fun_remove_TRUE = function(input, Min,start_index, Max, until,embedding_size_p, remove_threshold ){}
  
  fun_remove_FALSE = function(input, Min, start_index, Max, until, embedding_size_p){}
  
  streaming_score = function(input, Min = 100, start_index = start_index, Max = 0, until=0, batch = TRUE ,embedding_size_p = 0, remove=TRUE, remove_threshold = 0.2){
    total_start <- Sys.time()
    if(remove==TRUE){
      if(batch==TRUE){  # 
        pre=fun_batch_remove_TRUE(input=input, Min=Min, start_index= start_index, Max=Max, until=until, embedding_size_p=embedding_size_p, remove_threshold=remove_threshold )
      }else{
        pre=fun_remove_TRUE(input=input, Min=Min, start_index= start_index, Max=Max, until=until, embedding_size_p=embedding_size_p, remove_threshold=remove_threshold )
      }
    }else{
      if(batch==TRUE){
        pre=fun_batch_remove_FALSE(input=input, Min=Min, start_index=start_index, Max=Max, until=until, embedding_size_p=embedding_size_p )
      }else{
        pre=fun_remove_FALSE(input=input, Min=Min,start_index=start_index, Max=Max, until=until, embedding_size_p=embedding_size_p )
      }
    }
    total_end <- Sys.time()
    print(total_end - total_start)
    return(pre)
  }
}




#Result
{

  start_index  = which(cumsum(input$start) == 101)[1]
  last_index = nrow(input) 
  
  part = seq(start_index, last_index, 1000 )
  part = part[c((length(part)-4):(length(part)-1))]  
  output_total = data.frame()
  for(i in part){
    if(i != last_index){
      output = streaming_score(input, Min=100, start_index= i, Max=1000, until = 299, batch=TRUE, remove= FALSE, embedding_size_p=0)  # onehot
      if(is.null(output) == 0 ){
        output = output[order(output$timestamp),]
        start = min(which(output$leverage >=0))
        loc = which(output$leverage>=0)
        output = output[loc,]
        output_total = rbind(output_total, output)
      }
    }
  }
  setwd("/home/jonghyeon3/extension_AD/evaluations/total_result/memory_check_result")
  write.csv(output, "result_model1000_small.csv", row.names= FALSE)
}



# plot(see$leverage,  ylim= c(0,1),
#      col= ifelse(see$label==1 ,'red', 'black' ), cex= ifelse(see$label==1 ,1.0,  0.5), pch= ifelse(see$label==1 ,9,  1)
#      , ylab= 'Anomaly score')
# 
# plot(see2$leverage,  ylim= c(0,1),
#      col= ifelse(see2$label==1 ,'red', 'black' ), cex= ifelse(see2$label==1 ,1.0,  0.5), pch= ifelse(see2$label==1 ,9,  1)
#      , ylab= 'Anomaly score')


