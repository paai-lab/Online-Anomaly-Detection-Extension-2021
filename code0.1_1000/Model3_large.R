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


#data load and preprocess
{
input = data.frame(read.csv('large-0.1-1.csv', header=T))
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
  
  fun_batch_remove_TRUE = function(input, Min, start_index, Max, until, embedding_size_p, remove_threshold ){
    #prepare data
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
    
    if(start_index == last_index){
      #skip
    }else{
    
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
        prefixL[j] = nrow(cut)
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
        prefixL[j] = nrow(cut)
        save2 <- as.vector(t(cut))  
        newdat2[j,1:length(save2)] <- save2
      }
      newdat2[which(is.na(newdat2))] <- 0 # zero-padding
      newdat2_save= newdat2
      newdat3 = data.frame(cbind(Case=as.character(newdat[,1]), label= as.character(pre2$anomaly_type), newdat2))
      act_save = names(newdat) #change 1
      
      x2= newdat3[which(prefixL == prefixL[start_index]),-(1:2)]
      x2 = x2[,1:(prefixL[start_index]*num_act)]
    }
    
    #Caculate leverage
    x= as.matrix(sapply(x2, as.numeric))  
    h_diag <- fun_leverage(x)
    pre[start_index, 'leverage'] = h_diag[length(h_diag)]
    
    leverage_end <- Sys.time()
    
    pre[start_index, 'time'] =   (leverage_end-leverage_start)
    pre[start_index, 'tn'] = (h_diag[length(h_diag)] > (mean(h_diag)+sd(h_diag)))
    
    #Set escape option
    if(until==0){
      until = last_index
    }else{
      until= start_index+until
    }
    
    #Start event steam 
    remove_list = as.character()
    for(i in (start_index+1):until){ # last_index
      print(paste("Start to calculate leverage score of ", i ,"-th event (total ",event_num," events)" ,sep=''))
      leverage_start <- Sys.time()
      
      if(pre[(i-1),'leverage'] > remove_threshold ){
        remove_list = c(remove_list, pre[(i-1),'Case']) 
        pre2 = pre2[which(!is.element(pre2$Case,remove_list)),]
        prefixL = prefixL[which(!is.element(pre2$Case,remove_list))]
        newdat2_save = newdat2_save[which(!is.element(pre2[,1], remove_list)),]
      }
      
      if(is.element(pre[i,'Case'], remove_list)){
        print(paste( "(CaseID=",pre[i,'Case'],") was already detected as anomaly" ,sep=''))
        pre[i,'leverage'] = 1
      }else{
        pre2 = rbind(pre2, pre[i,])
        cur_len = sum(pre2$start)
        data<- pre2[,c("Case","Activity",'order')]  
        names(data)[1:2] <- c("ID", "ActivityID") 
        
        if(embedding_size_p>0){
          num_act= length(unique(data$ActivityID))
          embedding_size = round(num_act*embedding_size_p)
          
          # embedding encoding
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
          
          num_act= length(unique(data$ActivityID))
          
          { # update event
            newdat2<- matrix(NA, nrow=num_event , ncol=max) 
            prefixL = as.numeric()
            for(j in 1:num_event){
              cut = all3[which(all3[1:j,1]== all3[j,1] ),-c(1:2)]
              prefixL[j] = nrow(cut)
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
          c=unique(pre2[,c("Case","anomaly_type")])
          label = as.character(pre2[,c("anomaly_type")])
          
          if(sum(names(newdat)!=act_save)!=0){ #update event
            newdat2<- matrix(NA, nrow=num_event , ncol=max)  
            prefixL = as.numeric()
            for(j in 1:num_event){
              cut = newdat[which(newdat[1:j,1]== newdat[j,1] ),-1]
              prefixL[j] = nrow(cut)
              save2 <- as.vector(t(cut))  
              newdat2[j,1:length(save2)] <- save2
            }
          }else{ #update event
            newdat2<- matrix(NA, nrow=num_event , ncol=max)
            newdat2[1:nrow(newdat2_save), 1:min(max,ncol(newdat2_save))] = newdat2_save[,1:min(max,ncol(newdat2_save))]
            cut = newdat[which(newdat[,1]== object_case ),-1]
            prefixL = c(prefixL, nrow(cut))
            save2 <- as.vector(t(cut))  
            newdat2[nrow(newdat2_save)+1,1:length(save2)] <- save2
          }
          
          # Max option
          if(cur_len > Max ){
            del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
            del_case= del_case[which(!is.element(del_case, object_case))]
            pre2 = pre2[which(!is.element(newdat[,1], del_case)),]
            newdat2 = newdat2[which(!is.element(newdat[,1], del_case)),]
            label= label[which(!is.element(newdat[,1], del_case))]
            prefixL= prefixL[which(!is.element(newdat[,1], del_case))]
            newdat = newdat[which(!is.element(newdat[,1], del_case)),]
          }   
          
          newdat2[which(is.na(newdat2))] <- 0 # zero-padding
          newdat2_save= newdat2
          newdat3 <-data.frame(cbind(Case= as.character(newdat[,1]), label= label, newdat2))
          act_save = names(newdat) #change 1
          
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
      }
    }
    return(pre)
    }
  }
  
  fun_batch_remove_FALSE = function(input, Min,start_index, Max, until, embedding_size_p ){
    #prepare data
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
    
    if(start_index == last_index){
      #skip
    }else{
    
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
        prefixL[j] = nrow(cut)
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
        prefixL[j] = nrow(cut)
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
    pre[start_index, 'leverage'] = h_diag[length(h_diag)]
    
    leverage_end <- Sys.time()
    
    pre[start_index, 'time'] =   (leverage_end-leverage_start)
    pre[start_index, 'tn'] = (h_diag[length(h_diag)] > (mean(h_diag)+sd(h_diag)))
    
    #Set escape option
    if(until==0){
      until = last_index
    }else{
      until= start_index+until
    }
    
    #Start event steam 
    for(i in (start_index+1):until){ # last_index
      print(paste("Start to calculate leverage score of ", i ,"-th event (total ",event_num," events)" ,sep=''))
      leverage_start <- Sys.time()
      
      pre2 = rbind(pre2, pre[i,])
      cur_len = sum(pre2$start)
      data<- pre2[,c("Case","Activity",'order')]  
      names(data)[1:2] <- c("ID", "ActivityID") 
      
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
            prefixL[j] = nrow(cut)
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
        c=unique(pre2[,c("Case","anomaly_type")])
        label = as.character(pre2[,c("anomaly_type")])
        
        if(sum(names(newdat)!=act_save)!=0){ #update event
          newdat2<- matrix(NA, nrow=num_event , ncol=max)
          prefixL = as.numeric()
          for(j in 1:num_event){
            cut = newdat[which(newdat[1:j,1]== newdat[j,1] ),-1]
            prefixL[j] = nrow(cut)
            save2 <- as.vector(t(cut))  
            newdat2[j,1:length(save2)] <- save2
          }
        }else{ #update event
          newdat2<- matrix(NA, nrow=num_event , ncol=max)
          newdat2[1:nrow(newdat2_save), 1:min(max,ncol(newdat2_save))] = newdat2_save[,1:min(max,ncol(newdat2_save))]
          cut = newdat[which(newdat[,1]== object_case ),-1]
          prefixL = c(prefixL, nrow(cut))
          save2 <- as.vector(t(cut))  
          newdat2[num_event,1:length(save2)] <- save2
        }
        
        # Max option
        if(cur_len > Max ){
          del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
          del_case= del_case[which(!is.element(del_case, object_case))]
          pre2 = pre2[which(!is.element(newdat[,1], del_case)),]
          newdat2 = newdat2[which(!is.element(newdat[,1], del_case)),]
          label= label[which(!is.element(newdat[,1], del_case))]
          prefixL= prefixL[which(!is.element(newdat[,1], del_case))]
          newdat = newdat[which(!is.element(newdat[,1], del_case)),]
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
    }
    return(pre)
    }
  }
  
  fun_remove_TRUE = function(input, Min,start_index, Max, until,embedding_size_p, remove_threshold ){
    #prepare data
    pre<-input
    pre= pre[ with(pre, order(Case,timestamp)),]
    one= rep(1, nrow(pre))
    pre[,'start'] = ave(one, by= pre$Case, FUN= cumsum) -1
    pre[which(pre$start !=1),'start'] =0;pre= pre[ with(pre, order(timestamp)),]
    pre[,'Event'] = as.factor(1:nrow(pre));pre[,'num_case'] = cumsum(pre$start);pre[,'leverage'] = rep(-1, nrow(pre));pre[,'w-leverage'] = rep(-1, nrow(pre))
    pre[,'t1'] = rep(0, nrow(pre));pre[,'t2'] = rep(0, nrow(pre));pre[,'t3'] = rep(0, nrow(pre));pre[,'tn']= rep(0, nrow(pre));pre[,'w-tn']= rep(0, nrow(pre));pre[,'time'] = rep(0, nrow(pre))
    event_num = nrow(pre);case_num= length(unique(pre$Case))
    start_index  = start_index
    last_index = nrow(pre)

    leverage_start <- Sys.time()
    pre2 = pre[1:start_index,]
    cur_len = sum(pre2$start)
    data<- pre2[,c("Case","Activity","order")];names(data)[1:2] <- c("ID", "ActivityID") 
    
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
    
    if(start_index == last_index){
      #skip
    }else{
    
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
      
      newdat2<- matrix(NA, nrow=n , ncol=max)  
      for(j in 1:n){
        cut = all3[which(all3[,1]== c[j,1] ),-c(1:2)]
        save2 <- as.vector(t(cut))  
        newdat2[j,1:length(save2)] <- save2
      }
      newdat2[which(is.na(newdat2))] <- 0 # zero-padding
      newdat2_save= newdat2
      newdat3 = cbind(c, newdat2)
      # newdat3 = data.frame(cbind(Case=as.character(all3[,1]), label= as.character(pre2$anomaly_type), newdat2))
      x2= newdat3[,-(1:2)]
      
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
      
      newdat2<- matrix(NA, nrow=n , ncol=max)  
      for(j in 1:n){
        cut = newdat[which(newdat[,1]== c[j,1] ),-1]
        save2 <- as.vector(t(cut))  
        newdat2[j,1:length(save2)] <- save2
      }
      newdat2[which(is.na(newdat2))] <- 0 # zero-padding
      newdat2_save= newdat2
      l_save = c[,1]
      act_save = names(newdat) #change 1
      newdat3 = cbind(c, newdat2)
      # newdat3 = data.frame(cbind(Case=as.character(newdat[,1]), label= as.character(pre2$anomaly_type), newdat2))
      x2= newdat3[,-(1:2)]
    }
    
    #Caculate leverage
    x= as.matrix(sapply(x2, as.numeric))  
    h_diag <- fun_leverage(x)
    
    #Calculate weighted leverage
    length <- apply(x,1,sum)
    z_norm <- (length- mean(length))/sd(length)
    
    sigmoid_leng <- 1/(1+exp(-z_norm))
    if(embedding_size_p ==0 & (-2.2822+max(length)^0.3422 <0 | length(unique(length))==1) ){
      h_diag2 = h_diag}else{h_diag2 <-h_diag*(1-sigmoid_leng)^(-2.2822+max(length)^0.3422) } #weighted leverage
    h_diag2 = h_diag2*sum(h_diag)/sum(h_diag2)
    
    loc.case = which(c[,1]==object_case)
    pre[start_index, 'leverage'] = h_diag[loc.case]
    pre[start_index, 'w-leverage'] = h_diag2[loc.case]
    leverage_end <- Sys.time()
    
    pre[start_index, 'time'] =   (leverage_end-leverage_start)
    pre[start_index, 'tn'] = (h_diag[loc.case] > (mean(h_diag)+sd(h_diag)))
    pre[start_index, 'w-tn'] =  (h_diag2[loc.case] > (mean(h_diag2)+sd(h_diag2)))
    
    #Set escape option
    if(until==0){
      until = last_index
    }else{
      until= start_index+until
    }
    
    #Start event steam 
    remove_list = as.character()
    for(i in (start_index+1):until){ # last_index
      print(paste("Start to calculate leverage score of ", i ,"-th event (total ",event_num," events)" ,sep=''))
      leverage_start <- Sys.time()
      
      if(embedding_size_p == 0){
        if(pre[(i-1),'w-leverage'] > remove_threshold ){
          remove_list = c(remove_list, pre[(i-1),'Case'])
          pre2 = pre2[which(!is.element(pre2$Case,remove_list)),]
          l_save = l_save[which(!is.element(l_save,remove_list))]
          newdat2_save = newdat2_save[which(!is.element(l_save, remove_list)),]
        }
      }else{
        if(pre[(i-1),'leverage'] > remove_threshold ){
          remove_list = c(remove_list, pre[(i-1),'Case'])
          pre2 = pre2[which(!is.element(pre2$Case,remove_list)),]
          l_save = l_save[which(!is.element(l_save,remove_list))]
          newdat2_save = newdat2_save[which(!is.element(l_save, remove_list)),]
        }
      }
      
      if(is.element(pre[i,'Case'], remove_list)){
        print(paste( "(CaseID=",pre[i,'Case'],") was already detected as anomaly" ,sep=''))
        pre[i,'w-leverage'] = 1
        pre[i,'leverage'] = 1
        pre[i, 'time'] =   NA
        pre[i, 'tn'] = 1
        pre[i, 'w-tn'] = 1
      }else{
        pre2 = rbind(pre2, pre[i,])
        cur_len = sum(pre2$start)
        data<- pre2[,c("Case","Activity",'order')]  
        names(data)[1:2] <- c("ID", "ActivityID") 
        
        if(embedding_size_p>0){
          num_act= length(unique(data$ActivityID))
          embedding_size = round(num_act*embedding_size_p)
          
          # embedding encoding
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
          
          num_act= length(unique(data$ActivityID))
          
          { # update event
            newdat2<- matrix(NA, nrow=n , ncol=max)  
            for(j in 1:n){
              cut = all3[which(all3[,1]== c[j,1] ),-c(1:2)]
              save2 <- as.vector(t(cut))  
              newdat2[j,1:length(save2)] <- save2
            }
            loc = which(c[,1] == object_case)
          }
          
          # Max option
          if(cur_len > Max ){
            del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
            del_case= del_case[which(!is.element(del_case, object_case))]
            pre2 = pre2[which(!is.element(all3[,1], del_case)),]
            newdat2 = newdat2[which(!is.element(c[,1], del_case)),]
            label= label[which(!is.element(all3[,1], del_case))]
            all3 = all3[which(!is.element(all3[,1], del_case)),]
            c= c[which(!is.element(c[,1], del_case)),]
            loc = which(c[,1] == object_case)
          }   
          
          newdat2[which(is.na(newdat2))] <- 0 # zero-padding
          newdat2_save= newdat2
          l_save = c[,1]
          newdat3= cbind(c, newdat2)
          # newdat3 <-data.frame(cbind(Case= as.character(all3[,1]), label= label, newdat2))
          x2= newdat3[,-(1:2)]
          
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
          label = as.character(pre2[,c("anomaly_type")])
          
          { #update event
            if(sum(names(newdat)!=act_save)==0){
              newdat2<- matrix(NA, nrow=n , ncol=max)
              newdat2[1:nrow(newdat2_save), 1:min(max,ncol(newdat2_save))] = newdat2_save[,1:min(max,ncol(newdat2_save))]
              
              if(is.element(object_case, l_save) ){
                j = which(l_save == object_case)
                cut = newdat[which(newdat[,1]== l_save[j] ),-1]
                save2 <- as.vector(t(cut))
                newdat2[j,1:length(save2)] <- save2
                loc =j
              }else{
                j = which(c[,1] == object_case)
                cut = newdat[which(newdat[,1]== c[j,1] ),-1]
                save2 <- as.vector(t(cut))
                newdat2[n,1:length(save2)] <- save2
                loc = n
              }
            }else{
              newdat2<- matrix(NA, nrow=n , ncol=max)
              for(j in 1:n){
                cut = newdat[which(newdat[,1]== c[j,1] ),-1]
                save2 <- as.vector(t(cut))
                newdat2[j,1:length(save2)] <- save2
              }
              loc =  which(c[,1] == object_case)
            }
          }
          
          
          # Max option
          if(cur_len > Max ){
            del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
            del_case= del_case[which(!is.element(del_case, object_case))]
            pre2 = pre2[which(!is.element(newdat[,1], del_case)),]
            newdat2 = newdat2[which(!is.element(c[,1], del_case)),]
            label= label[which(!is.element(newdat[,1], del_case))]
            newdat = newdat[which(!is.element(newdat[,1], del_case)),]
            c= c[which(!is.element(c[,1], del_case)),]
            loc = which(c[,1] == object_case)
          }   
          
          newdat2[which(is.na(newdat2))] <- 0 # zero-padding
          newdat2_save= newdat2
          l_save = c[,1]
          act_save = names(newdat) #change 1
          newdat3= cbind(c, newdat2)
          # newdat3 <-data.frame(cbind(Case= as.character(newdat[,1]), label= label, newdat2))
          x2= newdat3[,-(1:2)]
        }
        
        #Calculate leverage
        x= as.matrix(sapply(x2, as.numeric))  
        h_diag <- fun_leverage(x)
        
        if(embedding_size_p==0){
          length <- apply(x,1,sum)
          z_norm <- (length- mean(length))/sd(length)
          sigmoid_leng <- 1/(1+exp(-z_norm))
          if(embedding_size_p ==0 &(-2.2822+max(length)^0.3422 <0 | length(unique(length))==1) ){
            h_diag2 = h_diag}else{h_diag2 <-h_diag*(1-sigmoid_leng)^(-2.2822+max(length)^0.3422) } #weighted leverage
          h_diag2 = h_diag2*sum(h_diag)/sum(h_diag2)
          pre[i, 'w-leverage'] = h_diag2[loc]
          pre[i, 'w-tn'] =  (h_diag2[loc] > (mean(h_diag2)+sd(h_diag2)))
        }
        
        pre[i, 'leverage'] = h_diag[loc]
        leverage_end <- Sys.time()
        
        print(paste("Anomaly score of", i ,"-th event = ", round( h_diag[loc],5), " (CaseID=",object_case,")"  ,sep=''))
        pre[i, 'time'] =   (leverage_end-leverage_start)
        pre[i, 'tn'] = (h_diag[loc] > (mean(h_diag)+sd(h_diag)))
      }
    }
    return(pre)
    }
  }
  
  fun_remove_FALSE = function(input, Min, start_index, Max, until, embedding_size_p){
    #prepare data
    pre<-input
    pre= pre[ with(pre, order(Case,timestamp)),]
    one= rep(1, nrow(pre))
    pre[,'start'] = ave(one, by= pre$Case, FUN= cumsum) -1
    pre[which(pre$start !=1),'start'] =0;pre= pre[ with(pre, order(timestamp)),]
    pre[,'Event'] = as.factor(1:nrow(pre));pre[,'num_case'] = cumsum(pre$start);pre[,'leverage'] = rep(-1, nrow(pre));pre[,'w-leverage'] = rep(-1, nrow(pre))
    pre[,'t1'] = rep(0, nrow(pre));pre[,'t2'] = rep(0, nrow(pre));pre[,'t3'] = rep(0, nrow(pre));pre[,'tn']= rep(0, nrow(pre));pre[,'w-tn']= rep(0, nrow(pre));pre[,'time'] = rep(0, nrow(pre))
    event_num = nrow(pre);case_num= length(unique(pre$Case))
    start_index  = start_index
    last_index = nrow(pre)

    leverage_start <- Sys.time()
    pre2 = pre[1:start_index,]
    cur_len = sum(pre2$start)
    data<- pre2[,c("Case","Activity","order")];names(data)[1:2] <- c("ID", "ActivityID") 
    
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
    
    if(start_index == last_index){
      #skip
      }else{
    
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
      
      newdat2<- matrix(NA, nrow=n , ncol=max)  
      for(j in 1:n){
        cut = all3[which(all3[,1]== c[j,1] ),-c(1:2)]
        save2 <- as.vector(t(cut))  
        newdat2[j,1:length(save2)] <- save2
      }
      newdat2[which(is.na(newdat2))] <- 0 # zero-padding
      newdat2_save= newdat2
      newdat3 = cbind(c, newdat2)
      # newdat3 = data.frame(cbind(Case=as.character(all3[,1]), label= as.character(pre2$anomaly_type), newdat2))
      x2= newdat3[,-(1:2)]
      
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
      
      newdat2<- matrix(NA, nrow=n , ncol=max)  
      for(j in 1:n){
        cut = newdat[which(newdat[,1]== c[j,1] ),-1]
        save2 <- as.vector(t(cut))  
        newdat2[j,1:length(save2)] <- save2
      }
      newdat2[which(is.na(newdat2))] <- 0 # zero-padding
      newdat2_save= newdat2
      l_save = c[,1]
      newdat3 = cbind(c, newdat2)
      # newdat3 = data.frame(cbind(Case=as.character(newdat[,1]), label= as.character(pre2$anomaly_type), newdat2))
      x2= newdat3[,-(1:2)]
      act_save = names(newdat) #change 1
    }
    
    #Caculate leverage
    x= as.matrix(sapply(x2, as.numeric))  
    h_diag <- fun_leverage(x)
    
    #Calculate weighted leverage
    length <- apply(x,1,sum)
    z_norm <- (length- mean(length))/sd(length)
    
    sigmoid_leng <- 1/(1+exp(-z_norm))
    if(embedding_size_p==0 & (-2.2822+max(length)^0.3422 <0 | length(unique(length))==1) ){
      h_diag2 = h_diag}else{h_diag2 <-h_diag*(1-sigmoid_leng)^(-2.2822+max(length)^0.3422) } #weighted leverage
    h_diag2 = h_diag2*sum(h_diag)/sum(h_diag2)
    
    loc.case = which(c[,1]==object_case)
    pre[start_index, 'leverage'] = h_diag[loc.case]
    pre[start_index, 'w-leverage'] = h_diag2[loc.case]
    leverage_end <- Sys.time()
    
    pre[start_index, 'time'] =   (leverage_end-leverage_start)
    pre[start_index, 'tn'] = (h_diag[loc.case] > (mean(h_diag)+sd(h_diag)))
    pre[start_index, 'w-tn'] =  (h_diag2[loc.case] > (mean(h_diag2)+sd(h_diag2)))
    
    #Set escape option
    if(until==0){
      until = last_index
    }else{
      until= start_index+until
    }
    
    #Start event steam 
    for(i in (start_index+1):until){ # last_index
      print(paste("Start to calculate leverage score of ", i ,"-th event (total ",event_num," events)" ,sep=''))
      leverage_start <- Sys.time()
      
      pre2 = rbind(pre2, pre[i,])
      cur_len = sum(pre2$start)
      data<- pre2[,c("Case","Activity",'order')]  
      names(data)[1:2] <- c("ID", "ActivityID") 
      
      if(embedding_size_p>0){
        num_act= length(unique(data$ActivityID))
        embedding_size = round(num_act*embedding_size_p)
        
        # embedding encoding
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
        
        num_act= length(unique(data$ActivityID))
        
        { # update event
          newdat2<- matrix(NA, nrow=n , ncol=max)  
          for(j in 1:n){
            cut = all3[which(all3[,1]== c[j,1] ),-c(1:2)]
            save2 <- as.vector(t(cut))
            newdat2[j,1:length(save2)] <- save2
          }
          loc = which(c[,1] == object_case)
        }
        
        # Max option
        if(cur_len > Max ){
          del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)]
          del_case= del_case[which(!is.element(del_case, object_case))]
          pre2 = pre2[which(!is.element(all3[,1], del_case)),]
          newdat2 = newdat2[which(!is.element(c[,1], del_case)),]
          label= label[which(!is.element(all3[,1], del_case))]
          all3 = all3[which(!is.element(all3[,1], del_case)),]
          c= c[which(!is.element(c[,1], del_case)),]
          loc = which(c[,1] == object_case)
        }
        
        newdat2[which(is.na(newdat2))] <- 0 # zero-padding
        newdat2_save= newdat2
        l_save = c[,1]
        newdat3= cbind(c, newdat2)
        # newdat3 <-data.frame(cbind(Case= as.character(all3[,1]), label= label, newdat2))
        x2= newdat3[,-(1:2)]
        
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
        label = as.character(pre2[,c("anomaly_type")])
        
        { #update event
          if(sum(names(newdat)!=act_save)==0){
            newdat2<- matrix(NA, nrow=n , ncol=max)
            newdat2[1:nrow(newdat2_save), 1:min(max,ncol(newdat2_save))] = newdat2_save[,1:min(max,ncol(newdat2_save))]
            
            if(is.element(object_case, l_save) ){
              j = which(l_save == object_case)
              cut = newdat[which(newdat[,1]== l_save[j] ),-1]
              save2 <- as.vector(t(cut))
              newdat2[j,1:length(save2)] <- save2
              loc =j
            }else{
              j = which(c[,1] == object_case)
              cut = newdat[which(newdat[,1]== c[j,1] ),-1]
              save2 <- as.vector(t(cut))
              newdat2[n,1:length(save2)] <- save2
              loc = n
            }
          }else{
            newdat2<- matrix(NA, nrow=n , ncol=max)
            for(j in 1:n){
              cut = newdat[which(newdat[,1]== c[j,1] ),-1]
              save2 <- as.vector(t(cut))
              newdat2[j,1:length(save2)] <- save2
            }
            loc =  which(c[,1] == object_case)
          }
        }
        
        
        # Max option
        if(cur_len > Max ){
          del_case = pre2[which(pre2$start==1),'Case'][1:(cur_len-Max)] 
          del_case= del_case[which(!is.element(del_case, object_case))]
          pre2 = pre2[which(!is.element(newdat[,1], del_case)),]
          newdat2 = newdat2[which(!is.element(c[,1], del_case)),]
          label= label[which(!is.element(newdat[,1], del_case))]
          newdat = newdat[which(!is.element(newdat[,1], del_case)),]
          c= c[which(!is.element(c[,1], del_case)),]
          loc = which(c[,1] == object_case)
        }   
        
        
        newdat2[which(is.na(newdat2))] <- 0 # zero-padding
        newdat2_save= newdat2
        l_save = c[,1]
        newdat3= cbind(c, newdat2)
        act_save = names(newdat) #change 1
        # newdat3 <-data.frame(cbind(Case= as.character(newdat[,1]), label= label, newdat2))
        x2= newdat3[,-(1:2)]
      }
      
      #Calculate leverage
      x= as.matrix(sapply(x2, as.numeric))  
      h_diag <- fun_leverage(x)
      
      #Calculate weighted leverage
      if(embedding_size_p==0){
        length <- apply(x,1,sum)
        z_norm <- (length- mean(length))/sd(length)
        sigmoid_leng <- 1/(1+exp(-z_norm))
        if(embedding_size_p == 0 &  (-2.2822+max(length)^0.3422 <0 | length(unique(length))==1) ){
          h_diag2 = h_diag}else{h_diag2 <-h_diag*(1-sigmoid_leng)^(-2.2822+max(length)^0.3422) } #weighted leverage
        h_diag2 = h_diag2*sum(h_diag)/sum(h_diag2)
        pre[i, 'w-leverage'] = h_diag2[loc]
        pre[i, 'w-tn'] =  (h_diag2[loc] > (mean(h_diag2)+sd(h_diag2)))
      }
      
      pre[i, 'leverage'] = h_diag[loc]
      leverage_end <- Sys.time()
      
      print(paste("Anomaly score of", i ,"-th event = ", round( h_diag[loc],5), " (CaseID=",object_case,")"  ,sep=''))
      pre[i, 'time'] =   (leverage_end-leverage_start)
      pre[i, 'tn'] = (h_diag[loc] > (mean(h_diag)+sd(h_diag)))
    }
    return(pre)
  }
  }
  
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
  part = part[-length(part)]  
  output_total = data.frame()
  for(i in part){
    output = streaming_score(input, Min=100, start_index= i, Max=1000, until = 99, batch=FALSE, remove= TRUE, embedding_size_p=0)  # onehot
    if(is.null(output) == 0 ){
      output = output[order(output$timestamp),]
      start = min(which(output$leverage >=0))
      loc = which(output$leverage>=0)
      output = output[loc,]
      output_total = rbind(output_total, output)
    }
  }
  setwd("/home/jonghyeon3/extension_AD/evaluations/result")
  write.csv(output_total, "result_model3_large.csv", row.names= FALSE)
}


