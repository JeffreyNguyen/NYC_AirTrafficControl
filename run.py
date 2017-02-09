import importlib
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from weather import *
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import sklearn.ensemble as ensemble

d_origin=load_data_all()

if __name__ == '__main__':
    lags=5
    degree=2
    training_months=6;
    d_all=d_origin.copy()
    d_all_p=process_data(d_all)
    lasso_param=.001
    
    #scale traffic for simulation
    global_factor=1 #maintain global traffic
    local_factor=0  #reduce local traffic to zero
    

    #randomly split data into testing and training
    df_train,df_test=df_rand_sample(d_all_p,n_months=training_months)
    
    #process datasets for regression, this includes 2nd order interactions
    train_in,train_out,in_rows,labels=make_training_data(df_train,lags=lags,degree=degree)
    test_in,test_out,out_rows,labels=make_training_data(df_test,lags=lags,degree=degree)
    
    #process datasets for regression, this only has first order interactions
    nntrain_in,nntrain_out,nnin_rows,nnlabels=make_training_data(df_train,lags=lags,degree=1)
    nntest_in,nntest_out,nnout_rows,nnlabels=make_training_data(df_test,lags=lags,degree=1)
    
    #train models, and produce metrics
    nn=neural_model(nntrain_in,nntrain_out,layers=(10,20,5))
    nntrain_mae=metrics.mean_absolute_error(np.exp(nntrain_out),np.exp(nn.predict(nntrain_in)))
    nntest_mae=metrics.mean_absolute_error(np.exp(nntest_out),np.exp(nn.predict(nntest_in)))
    
    reg = linear_model.Lasso(alpha=lasso_param)
    reg.fit(train_in,train_out)
    train_mae=metrics.mean_absolute_error(np.exp(train_out),np.exp(reg.predict(train_in)))
    test_mae=metrics.mean_absolute_error(np.exp(test_out),np.exp(reg.predict(test_in)))
    
    print train_mae,test_mae,reg.score(train_in,train_out),reg.score(test_in,test_out)
    print nntrain_mae,nntest_mae,nn.score(nntrain_in,nntrain_out),nn.score(nntest_in,nntest_out)
    
    #run simulation with reduced traffic to see the reduction in pollution
    run_experiment(df,reg,local_factor=local_factor,global_factor=global_factor,lags=lags,degree=degree)
    
    