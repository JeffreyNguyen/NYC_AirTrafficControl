import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.signal as sig
import scipy.stats as stats
from sklearn import linear_model
import timeit
from geopy.geocoders import Nominatim
from flask import app
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import operator


# load lat/lon for each measurement station
def get_locations():
    locations=[] #initialize list for dataframes
    location_strings=[]
    geolocator = Nominatim()

    
    for iFile in range(1,16):
        #datasets are numbered as airX.csv
        fname = 'Datasets/air' + str(iFile) + '.csv'
        #load first row as pd.dataframe object
        df = pd.read_csv(fname, header=0,nrows=1) 
        #get coordinates
        df = df[['lat','lon']]
        #set file ID
        df['id']=iFile
        
        coord_string=str(df.iloc[0].lat) +', '+ str(df.iloc[0].lon)
        #display location, requires internet connection
        location_string=geolocator.reverse(coord_string)
        
        location_strings+=[location_string]
        locations+=[df]

    locations=pd.concat(locations)
    return locations

#find nodes that are within some distance of the locations in d_loc, given by get_locations
def get_nodes(d_loc): 
    close_threshold=0.005
    #load sets of nodes
    fname = 'Datasets/nodes.csv'
    d_nodes = pd.read_csv(fname, header=0)
    #xcoord and ycoord are lat and lon respectively
    d_nodes = d_nodes[['node_id','xcoord','ycoord']]
    
    #get node coordinates
    nodes_lon=np.array(d_nodes['xcoord'].as_matrix())[np.newaxis].T
    nodes_lat=np.array(d_nodes['ycoord'].as_matrix())[np.newaxis].T
    #get input coordinates
    cent_lat=np.array(locations['lat'].as_matrix())[np.newaxis]
    cent_lon=np.array(locations['lon'].as_matrix())[np.newaxis]
    
    #calculate distance between nodes and input locations (cityblock)
    #select nodes closer than some theshold
    close_lon=np.absolute(nodes_lon-cent_lon) < close_threshold
    close_lat=np.absolute(nodes_lat-cent_lat) < close_threshold
    
    #do and operation between close in lat, and close in lon
    close=close_lon+close_lat
    close=close.any(axis=1)
    
    #add closeness to nodes dataframe, so each node as a new column "n" 
    #and in n, 1 indicates the nodes is close to location n
    for i_cent in range(dmat.shape[1]):
        d_nodes[str(i_cent+1)]=dmat[:,i_cent]  
    print 'Number of nodes:', d_nodes.size
    d_nodes=d_nodes[close==True]
    
    return d_nodes

def get_traffic(d_nodes):
    # get sum of weights for nodes close to locations. uses output of get_nodes()
    # this code can be slow! takes a long time to process all the large datasets. 
    #list of traffic data
    fnames = ['Datasets/travel_times_2010.csv',
             'Datasets/travel_times_2011.csv',
             'Datasets/travel_times_2012.csv',
             'Datasets/travel_times_2013.csv']
    d_traffic=[]
    
    for fname in fnames:
        file_out=[]
        file_size=sum(1 for line in open(fname)) *1.0
        print file_size
        completed=0.0
        chunk_size=1000000;
        #load chunk of data in fname
        for chunk in pd.read_csv(fname, chunksize=chunk_size):
            
            chunk_out=[]
            
            #convert dataframe to  datetime series
            chunk['datetime']=pd.to_datetime(
                chunk['datetime'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
            chunk = chunk.set_index(pd.DatetimeIndex(chunk['datetime']))
            
            #do grouping by proximity to relavent nodes
            #loop over the locations
            for i_loc in range(d_nodes.shape[1]-3):
                #select nodes close to current location
                node_list=d_nodes['node_id'].loc[d_nodes[str(i_loc+1)]==True]
                
                #select links that have both end nodes close to 
                overlap = chunk['begin_node_id'].isin(node_list)
                temp=chunk[overlap==True]
                overlap = temp['end_node_id'].isin(node_list)
                temp=temp[overlap==True]

                #groupby datetime (1 hr resolution)
                g=temp.groupby([temp.index.values])
                output=pd.DataFrame({str(i_loc+1):g['travel_time'].sum()})
                chunk_out=chunk_out + [output]
            #combine results for each location
            file_out=file_out+[pd.concat(chunk_out,axis=1)]
            completed=completed+chunk_size;
            sys.stdout.write('\rCompleted '+str(completed/file_size))
        #combine results for each chunk
        d_traffic=d_traffic+[pd.concat(file_out)]
    #combine results for each file
    d_traffic=pd.concat(d_traffic)
    d_traffic=d_traffic.groupby([d_traffic.index.values]).sum()
    return d_traffic


#load air quality datasets and dataframe, also return coordinates
def read_air_quality(fileID=6):
    fname = 'Datasets/air' + str(fileID) + '.csv'
    df = pd.read_csv(fname, header=0)
    
    first_row=df.iloc[0]
    location=first_row[['lat','lon']]
    
    #load realvent data and set to datetime index. 
    df=df[['datetime','value']]
    df['datetime'] =  pd.to_datetime(df['datetime'], format='%Y%m%dT%H',errors='coerce')
    df = df.set_index(pd.DatetimeIndex(df['datetime']))
    return df,location
    
#Simple groupby date for pollution value, also gets daily sdev      
def course_grain(df,timebin='d'):
    g = df.groupby(pd.TimeGrouper('d')) 
    dg=pd.DataFrame({'value':g['value'].max(),
                     'var':g['value'].std()})
    return dg

#Simple time lag adder, no longer used
def add_lags(df):
    df_lag=df[["value","var","isspike"]]
    d_lag1=df_lag.shift(-1)
    d_lag1.columns=["value_lag1","var_lag1","isspike_lag1"]
    d_lag2=df_lag.shift(-2)
    d_lag2.columns=["value_lag2","var_lag2","isspike_lag2"]
    df=pd.concat([df, d_lag1, d_lag2], axis=1)
    return df
        
def spike_detection(df,thresh=20):
#find spikes in the pollution trace, stored in 'value'

    value=df['value'].as_matrix()
    
    #do peak detection
    spike1= sig.find_peaks_cwt(value,np.arange(1,3),gap_thresh=1)
    
    # appply additional threshold which is max of thresh and the mean. 
    thresh=np.max([value.mean(),thresh])
    spike2=np.where(value>thresh)
    spike=np.intersect1d(spike1,spike2)
    
    #make isspike data a column, 
    isspike=np.zeros(value.size)
    isspike[spike]=1
    df['isspike']=isspike
    return df

def count_spikes(df,timebin='M'):
    g = df.groupby(pd.TimeGrouper(timebin)) 
    dg=pd.DataFrame({'value':g['value'].mean(),
                     'var':g['value'].std(),
                     'spikecount':g['isspike'].sum()})
    return dg

def read_oil(fileID=1):
    #load csv with heating oil data, has borough, month, and total oil used for many buildings
    #The output is specific to the location specified by fileID, matching the air quality reading location
    fname = 'Datasets/Heating_Oil_Consumption.csv'
    df = pd.read_csv(fname, header=0)
    df=df[['Borough','Revenue Month',
           '# days','Consumption']]
    
    #get daily average of heating oil consumption
    df['Consumption_Oil']=df['Consumption'].astype(float)/df['# days'].astype(float)
    
    #group by month and location
    dg=df.groupby(['Revenue Month','Borough'], as_index=False).sum()
    dg['Revenue Month'] =  pd.to_datetime(dg['Revenue Month'], format='%m/%d/%y')
    dg=dg.sort_values(by='Revenue Month',ascending=True).reset_index()

    #hand coded correspondence between locations ID's and boroughs
    if fileID<=8:
        dg2=dg[dg['Borough'] =='MANHATTAN']
    elif fileID<=11:
        dg2=dg[dg['Borough'] =='BRONX']
    elif fileID<=13:
        dg2=dg[dg['Borough'] =='QUEENS']
    elif fileID<=15:
        dg2=dg[dg['Borough'] =='BROOKLYN']
        
    dg2['Revenue Month'] =  pd.to_datetime(dg2['Revenue Month'])
    return dg2

def read_gas(fileID=1):
    #load csv with heating gas data, has borough, month, and total oil used for many buildings
    #The output is specific to the location specified by fileID, matching the air quality reading location

    fname = 'Datasets/Heating_Gas_Consumption.csv'
    df = pd.read_csv(fname, header=0) #outputs pd.dataframe object
    df=df[['Borough','Revenue Month',
           '# days','Consumption']]
    #get daily average of heating oil consumption
    df['Consumption_Gas']=df['Consumption'].astype(float)/df['# days'].astype(float)
    #group by month and location
    dg=df.groupby(['Revenue Month','Borough'], as_index=False).sum()
    dg['Revenue Month'] =  pd.to_datetime(dg['Revenue Month'], format='%m/%d/%y')
    dg=dg.sort_values(by='Revenue Month',ascending=True).reset_index()
    
    #hand coded correspondence between locations ID's and boroughs
    if fileID<=8:
        dg2=dg[dg['Borough'] =='MANHATTAN']
    elif fileID<=11:
        dg2=dg[dg['Borough'] =='BRONX']
    elif fileID<=13:
        dg2=dg[dg['Borough'] =='QUEENS']
    elif fileID<=15:
        dg2=dg[dg['Borough'] =='BROOKLYN']
        
    dg2['Revenue Month'] =  pd.to_datetime(dg2['Revenue Month'])
    return dg2


def read_energy(fileID=1):
    #load both heating oil and heating gas usage around a single air sensor location
    dgas=read_gas(fileID)
    doil=read_oil(fileID)
    denergy=pd.merge(doil,dgas, on=['Borough','Revenue Month'])

    denergy=denergy[['Revenue Month',
                     'Borough',
                     'Consumption_Oil',
                     'Consumption_Gas']]
    denergy2=denergy.groupby(['Revenue Month']).sum()
    #resample to 1d resolution with interpolation
    denergy2=denergy2.resample('1d').sum()
    denergy2=denergy2.interpolate()

    return denergy2


def read_weather():
    #load all weather data, also getting location
    fname = 'Datasets/nyc_weather.csv'
    df = pd.read_csv(fname, header=0)
    
    #pase location, there's only one location, its nyc...
    first_row=df.iloc[0]
    coordinate=first_row[['LATITUDE','LONGITUDE']]
    
    # only get FM-15 reports, others have things we don't want
    df=df[df['REPORTTYPE']=='FM-15']
    
    df=df[['DATE','HOURLYWETBULBTEMPC',
           'HOURLYDRYBULBTEMPC','HOURLYDewPointTempC',
          'HOURLYWindSpeed','HOURLYWindDirection',
          'HOURLYRelativeHumidity','HOURLYPrecip']]
    #clean up wind data, wind angle to wind_x and wind_y
    df.loc[df['HOURLYWindDirection'] =='VRB','HOURLYWindDirection'] = np.nan
    df['wind_y']=np.sin(np.deg2rad(df['HOURLYWindDirection'].astype(float)))
    df['wind_x']=np.cos(np.deg2rad(df['HOURLYWindDirection'].astype(float)))
    df.loc[np.isnan(df['wind_x']),['wind_x','wind_y']] = 0
    
    #clean up temp reading
    df['HOURLYDRYBULBTEMPC']=df['HOURLYDRYBULBTEMPC'].astype(str)
    df['HOURLYDRYBULBTEMPC'] = df['HOURLYDRYBULBTEMPC'].map(lambda x: x.rstrip('s'))
    df['HOURLYDRYBULBTEMPC']=df['HOURLYDRYBULBTEMPC'].astype(float)
    
    #clean up precip reading
    df.loc[df['HOURLYPrecip'] =='T','HOURLYPrecip'] = 0
    df['HOURLYPrecip']=df['HOURLYPrecip'].astype(str)
    df['HOURLYPrecip'] = df['HOURLYPrecip'].map(lambda x: x.rstrip('s'))
    df['HOURLYPrecip']=df['HOURLYPrecip'].astype(float)
    df.loc[np.isnan(df['HOURLYPrecip']),'HOURLYPrecip'] = 0

    #format datetime
    df['DATE'] =  pd.to_datetime(df['DATE'], format='%m/%d/%y %H:%M')
    df = df.set_index(pd.DatetimeIndex(df['DATE']))

    return df

def coursegrain_weather(df,timebin='d'):
    g = df.groupby(pd.TimeGrouper(timebin)) 
    dg=pd.DataFrame({'TempAvg':g['HOURLYDRYBULBTEMPC'].mean(),
                     'TempMax':g['HOURLYDRYBULBTEMPC'].max(),
                     'TempMin':g['HOURLYDRYBULBTEMPC'].min(),
                     'Wind_y':g['wind_y'].mean(),
                     'Wind_x':g['wind_x'].mean(),
                     'Wind_v':g['HOURLYWindSpeed'].mean(),
                     'Precip':g['HOURLYPrecip'].sum(),
                     'Humidity':g['HOURLYRelativeHumidity'].mean()})
    return dg

def process_data(df):
    #process data before fitting regression or NN, some for zscore,  
    z_features=['Humidity',   'TempAvg',  'TempMax',  'TempMin',
                'Consumption_Gas','Consumption_Oil']
    
    for feature in z_features:
        df[feature]=(df[feature]-df[feature].mean())/df[feature].std()
        
    scale_features = ['var','Wind_v']
    for feature in scale_features:
        df[feature]=(df[feature]/df[feature].mean())
        
    log_features = ['value']
    for feature in log_features:
        df[feature]=np.log(df[feature])
    return df

def make_traffic_input(df,location=1):
    df['location']=location
    df['local_traffic']=0
    input_data=make_traffic_trainer(df)[0]
    return input_data

    print df.head(4)
    bin_data=np.array([df.index.dayofweek, df.index.month-1,df['location']-1]).T
    enc=preprocessing.OneHotEncoder(n_values=[7,12,15])
    enc.fit(bin_data)
    bin_data=enc.transform(bin_data).toarray()
        
    poly = preprocessing.PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
    cross_features=['Humidity',   'TempAvg',  'TempMax',  'TempMin','Wind_v','Precip']

    #start with more static data
    input_data=df[cross_features].as_matrix()
    input_data=poly.fit_transform(input_data)
    input_data=np.concatenate((input_data,bin_data),axis=1)
    return input_data
    
def make_traffic_trainer(df):
    print df.head(4)
    bin_data=np.array([df.index.dayofweek, df.index.month-1,df['location']-1]).T
    enc=preprocessing.OneHotEncoder(n_values=[7,12,15])
    enc.fit(bin_data)
    bin_data=enc.transform(bin_data).toarray()
    
    poly = preprocessing.PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
    cross_features=['Humidity',   'TempAvg',  'TempMax',  'TempMin','Wind_v','Precip']

    #start with more static data
    input_data=df[cross_features].as_matrix()
    input_data=poly.fit_transform(input_data)
    input_data=np.concatenate((input_data,bin_data),axis=1)
#    good_col=-np.isnan(input_data).all(axis=0)
#    input_data=input_data[:,good_col]
    return input_data,output_data

    
def predict_traffic():
    return 
    


def split_test_train(input_data,output_data,train=.8):
    n_points=output_data.shape[0]
    n_train=n_points*train
    n_test=n_points-n_train
    train_input=input_data[:n_train,:]
    train_output=output_data[:n_train]
    test_input=input_data[n_train:,:]
    test_output=output_data[n_train:] 
    
    return train_input,train_output,test_input,test_output

def df_rand_sample(df,n_months=4):
    #make random sample of months for cross validation
    years=df.index.year
    years=years-min(years)
    month_index=years*12+df.index.month-1
    month_select=np.random.choice(48,n_months,replace=False)
    selected=[np.any(x==month_select) for x in month_index]
    not_selected=[np.all(x!=month_select) for x in month_index]
    df_test=df.ix[selected]
    df_train=df.ix[not_selected]
    return df_train,df_test
    
    
def make_training_data(d_in,lags=1,degree=2):
    #make training and testing splits from dataframe for fitting, also generates timelags and polynomial features
    df=d_in.copy()
    
    #features to use for fitting, only some with have lags
    lag_features=['value','var','Humidity', 
                  'TempMax',  'TempMin','Wind_v', 
                  'Wind_x','Precip','Wind_y','local_traffic','total_traffic']
    features=['Consumption_Oil','Consumption_Gas']
    
    #Binary data features
    days = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    months= ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'];
    locations=['loc'+str(x) for x in range(1,16)]
    
    #get binary data as day of week, month of year, and location index
    bin_data=np.array([df.index.dayofweek, df.index.month-1, df['location']-1]).T
    bin_data_label=days+months+locations
    
    # convert to binary features
    enc=preprocessing.OneHotEncoder(n_values=[7, 12, 15])
    enc.fit(bin_data)
    bin_data=enc.transform(bin_data).toarray()

    
    poly = preprocessing.PolynomialFeatures(degree,interaction_only=True,include_bias=False)
    
    #start with more static data
    input_data=df[features].as_matrix()
    
    #add todays features, ignoring info about pollution
    lag_data0=df[lag_features[2:]].as_matrix()
    
    #add lag features and higher order lag features
    lag_data0_=poly.fit_transform(lag_data0)
    
    #combine all data before lags
    input_data=np.concatenate((lag_data0_,input_data,bin_data),axis=1)

    #construct datalabels 
    cross_features=polynomial_labels(lag_features[2:],degree)
    cross_features_lag=polynomial_labels(lag_features,degree)
    data_labels=cross_features+features+bin_data_label
    
    
    #add time lags
    location_list=df['location'].unique()
    for i_lag in range(1,lags+1):
        df_temp=df.copy()
        
        #have to be careful about adding time lags, we go through each location
        for location in location_list:
            #seperate current location from others
            df_loc=df_temp[df_temp['location']==location]
            df_not_loc=df_temp[df_temp['location']!=location]
            
            #shift index for current location, and then shift index back 
            df_loc.index=df_loc.index- pd.DateOffset(1)
            df_loc=df_loc.ix[df_loc.index +pd.DateOffset(1)]
            #fix location after shifting
            df_loc['location']=location
            #add back current location
            df_temp=df_not_loc.append(df_loc)
        
        #poly transform fails with nans
        #temporarily fill nans with -999, get poly features,  then remove them later
        df_temp=df_temp.fillna(-999)
        lag_data0=df_temp[lag_features].as_matrix()
        lag_data0_=poly.fit_transform(lag_data0)
        lag_data0_[lag_data0_==-999]=np.nan
        
        #add lag features to current input data
        input_data=np.concatenate((lag_data0_,input_data),axis=1)
        
        #add labels lags
        lag_label=[x+'lag'+str(i_lag) for x in cross_features_lag]
        data_labels=lag_label+data_labels
    
    #get all labels
    labels=df.columns;

    #kill off nans
    output_data=df['value'].as_matrix()
    good_row=-np.isnan(input_data).any(axis=1)
    input_data=input_data[good_row,:]
    
    #remove bad rows, but keep track of where we did it
    output_data=output_data[good_row]
    return input_data,output_data,good_row,data_labels


def polynomial_labels(labels,degree):
    #take labels for features and return higher order pairs of labels, use to get labels after getting polynomial features
    output_label=list(labels)
    if degree==2:
        for i,name_i in enumerate(labels):
            for name_j in labels[i+1:]:
                next_label=name_i +' x ' + name_j
                output_label+=[next_label]
    return output_label



def regression_model(input_data,output_data,alpha=0.0001,l1_ratio=0.00005):
    #reg = linear_model.LinearRegression()
    reg = linear_model.ElasticNet(alpha, l1_ratio)
    reg.fit (input_data,output_data)
    model_out=reg.predict(input_data)
    return reg

def neural_model(input_data,output_data,layers=(15,5)):
    #reg = linear_model.LinearRegression()
    nn = MLPRegressor(layers)
    nn.fit (input_data,output_data)
    model_out=nn.predict(input_data)
    return nn

def get_coefs(labels,coef):
    relavent_locations=np.where(np.absolute(np.array(coef))>.01)[0]
    relavent_values=coef[relavent_locations]
    relavent_keys=[labels[i] for i in relavent_locations]
    x=dict(zip(relavent_keys,relavent_values))
    sorted_x = dict(sorted(x.items(), key=operator.itemgetter(1),reverse=True))
    return sorted_x

def read_traffic():
    fname = 'Datasets/traffic_summary.csv'
    df = pd.read_csv(fname, header=0,index_col=0) #outputs pd.dataframe object
    df.index = pd.to_datetime(df.index)
    df = df.groupby(pd.TimeGrouper('d')).sum()
    cols=list(df.columns)
    df['total_traffic']=df.mean(axis=1)
    return df

def read_traffic_alt():
    fname = 'Datasets/traffic_summary.csv'
    df = pd.read_csv(fname, header=0,index_col=0) #outputs pd.dataframe object
    df.index = pd.to_datetime(df.index)
    df=df.apply(traffic_normalize)
    
    df = df.groupby(pd.TimeGrouper('d')).mean()
    
    d_out=pd.DataFrame()
    
    for col in list(df.columns):
        d_temp=pd.DataFrame(df[col])
        
        d_temp['location']=[float(col)]*d_temp.shape[0]
        d_temp.columns=['local_traffic','location']
        d_temp['total_traffic']=df.mean(axis=1)

        d_out=pd.concat((d_out,d_temp))
        
    return d_out

def traffic_normalize(values_in):
    #normalize traffic data by the lower 20percentile value
    values=values_in[-np.isnan(values_in)]
    hist_out,x=np.histogram(values,50)
    hist_out=1.0*hist_out.cumsum()
    hist_out=hist_out/hist_out.max()
    low=x[np.where(hist_out>.20)[0][0]]
    
    values_out=values_in/low-1
    values_out[values_out<0]=0
    return(values_out)



def load_data(loc_index=1):
    d_all,location = read_air_quality(loc_index)
    d_all = course_grain(d_all,'d')
    d_all = spike_detection(d_all,30)
    d_all = add_lags(d_all)
    
    d_weather = read_weather()
    d_weather = coursegrain_weather(d_weather)
    
    d_energy = read_energy(loc_index)
    d_traffic= read_traffic()
    
    #merge all datasets on datetime index
    d_all=d_all.merge(d_weather,how='outer',left_index=True, right_index=True)
    d_all=d_all.merge(d_energy,how='outer',left_index=True, right_index=True)
    d_all=d_all.merge(d_traffic,how='outer',left_index=True, right_index=True)
    good_time =-d_all.isnull().values.any(axis=1)
    d_all=d_all[good_time==True]
    return d_all,location

def load_data_all():
    #load all data for weather, pollution, energy useage, and traffic
    d_weather = read_weather()
    d_weather = coursegrain_weather(d_weather)
    d_traffic= read_traffic_alt()
    d_all=[];
    #loop through air quality files
    for fileID in range(1,16):
        d_file,location = read_air_quality(fileID)
        d_file = course_grain(d_file,'d')
        d_file = spike_detection(d_file,30)
        
        d_energy = read_energy(fileID)
        
        #add weather, energy, and traffic for each location
        d_file=d_file.merge(d_energy,how='outer',left_index=True, right_index=True)
        d_file=d_file.merge(d_weather,how='outer',left_index=True, right_index=True)
        
        d_traffic_ID=d_traffic[d_traffic['location']==fileID]
        d_file=d_file.merge(d_traffic_ID,how='outer',left_index=True, right_index=True)
        
        #remove nan rows
        good_time =-d_file.isnull().values.any(axis=1)
        d_file=d_file[good_time==True]
        
        #combine all data
        d_all=d_all+[d_file]
        
    d_all=pd.concat(d_all)
        
    return d_all

def fit_models(d_origin,):
    lags=5
    degree=2
    d_origin=load_data_all()
    d_all=d_origin.copy()

    d_all_p=process_data(d_all)
    df_train,df_test=df_rand_sample(d_all_p,n_months=12)
    train_in,train_out,in_rows,labels=make_training_data(df_train,lags=lags,degree=degree)
    test_in,test_out,out_rows,labels=make_training_data(df_test,lags=lags,degree=degree)

    nn=neural_model(train_in,train_out,layers=(10,10,5))

    #nntrain_mae=metrics.mean_absolute_error(np.exp(train_out),np.exp(nn.predict(train_in)))
    #nntest_mae=metrics.mean_absolute_error(np.exp(test_out),np.exp(nn.predict(test_in)))
    reg = linear_model.Lasso(alpha=.0001)
    #reg.fit(train_in,train_out)
    #train_mae=metrics.mean_absolute_error(np.exp(train_out),np.exp(reg.predict(train_in)))
    #test_mae=metrics.mean_absolute_error(np.exp(test_out),np.exp(reg.predict(test_in)))
    #print train_mae,test_mae,reg.score(train_in,train_out),reg.score(test_in,test_out)
    return reg,nn, train_in, train_out,test_in,test,out

def run_experiment(df,reg,local_factor=1,global_factor=1,lags=5,degree=2):
    #run model with simulated traffic levels, adding column for current prediction, simulated result, and difference
    
    d_exp_all=pd.DataFrame()
    
    #run simulation at each location
    for location in range(1,15):
        d_exp=df.copy()
        d_exp2=df.copy()
        d_exp=process_data(d_exp)
        try:
            d_exp=d_exp[d_exp['location']==location]
            d_exp2=d_exp2[d_exp2['location']==location]
            
            # get model inputs
            predicted_in,__,good_row,__=make_training_data(d_exp,lags=lags,degree=degree)
            
            #simulate reduced traffic and get model inputs
            d_exp['local_traffic']=d_exp['local_traffic']*local_factor
            d_exp['total_traffic']=d_exp['total_traffic']*global_factor
            experiment_in,__,good_row,labels=make_training_data(d_exp,lags=lags,degree=degree)

            #clean up shape by indexing
            predicted_out=np.zeros(good_row.shape[0])*np.nan
            experiment_out=np.zeros(good_row.shape[0])*np.nan

            #apply models, and dont for get to undo log transform
            predicted_out[good_row]=np.exp(reg.predict(predicted_in))
            experiment_out[good_row]=np.exp(reg.predict(experiment_in))

            #add results to datastructure
            d_exp2['predicted']=predicted_out
            d_exp2['exp_value']=experiment_out
            d_exp2['reduction']=predicted_out-experiment_out
            d_exp2['error']=d_exp2['value']-predicted_out
            
            d_exp_all=d_exp_all.append(d_exp2)
    
        except ValueError:
            print ValueError
    return d_exp_all
