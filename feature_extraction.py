import json 
import boto3
import csv
import ast

import logging
from botocore.exceptions import ClientError
import os

import pandas as pd
import numpy as np 


s3 = boto3.client('s3')
#s3 = boto3.resource('s3') 

def lambda_handler(event, context):
    #print(event) 
    bucket = event['Records'][0]['s3']['bucket']['name']
    json_file_name = event['Records'][0]['s3']['object']['key']    
    print(bucket) 
    print(json_file_name)
        

    obj = s3.get_object(Bucket = bucket, Key = json_file_name) 
 
    body = obj['Body'].read().decode().split('\t')  
    
    print('oi') 
    #print(type(body))
    data = pd.DataFrame(columns=[ "P_PDG",	"P_TPT",	"T_TPT",	"P_MON_CKP",	"T_JUS_CKP",	"P_JUS_CKGL",	"T_JUS_CKGL",	"QGL"])
    count = len(body)-1
    for i in range(count):
        aux = body[i]
        objeto = json.loads(aux) 
        data = data.append(objeto, ignore_index=True)  
 
     
    ################################################################################################################
    # Normalização (verificar a precisão)! 
    ################################################################################################################
    
    def mean_norm(df_input):
        return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    
    # NAN -> 0
    data_n = mean_norm(data)
    data_n = data_n.fillna(0)
    
    #print(data_n) 
    #return data_n

  
    ################################################################################################################
    # FUNCTION
    ################################################################################################################
    # 1)
    print('Function')
    def abs_energy(x):
        """
        Returns the absolute energy of the time series which is the sum over the squared values
    
        .. math::
    
            E = \\sum_{i=1,\\ldots, n} x_i^2
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, (np.ndarray, pd.Series)):
            x = np.asarray(x)
        return np.dot(x, x)
    
    # 2)
    
    def sum_values(x):
        """
        Calculates the sum over the time series values
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if len(x) == 0:
            return 0
    
        return np.sum(x)
    
    # 3)
    
    def median(x):
        """
        Returns the median of x
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.median(x)
    
    # 4) 
    
    def mean(x):
        """
        Returns the mean of x
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.mean(x)
    
    # 5) 
    
    def length(x):
        """
        Returns the length of x
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: int
        """
        return len(x)
    
    # 6)
    
    def standard_deviation(x):
        """
        Returns the standard deviation of x
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.std(x)
    
    # 7)
    
    def variance(x):
        """
        Returns the variance of x
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.var(x)
        
    # 8)
    
    def root_mean_square(x):
        """
        Returns the root mean square (rms) of the time series.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.NaN
    
    # 9)
    
    def maximum(x):
        """
        Calculates the highest value of the time series x.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.max(x)
    
    # 10)
    
    def minimum(x):
        """
        Calculates the lowest value of the time series x.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.min(x)
    
    # 11)
    
    def mean_abs_change(x):
        """
        Average over first differences.
    
        Returns the mean over the absolute differences between subsequent time series values which is
    
        .. math::
    
            \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|
    
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        return np.mean(np.abs(np.diff(x)))
        
    # 12) 
    
    def mean_change(x):
        """
        Average over time series differences.
    
        Returns the mean over the differences between subsequent time series values which is
    
        .. math::
    
            \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN
    
    # 13)
    
    def mean_second_derivative_central(x):
        """
        Returns the mean value of a central approximation of the second derivative
    
        .. math::
    
            \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        x = np.asarray(x)
        return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN
    
    # 14)
    
    def variation_coefficient(x):
        """
        Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        mean = np.mean(x)
        if mean != 0:
            return np.std(x) / mean
        else:
            return np.nan
            
    # 15)
    
    def skewness(x):
        """
        Returns the sample skewness of x (calculated with the adjusted Fisher-Pearson standardized
        moment coefficient G1).
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return pd.Series.skew(x)
    
    # 16) 
    
    def kurtosis(x):
        """
        Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
        moment coefficient G2).
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        return pd.Series.kurtosis(x)
        
    # 17)
    
    def percentage_of_reoccurring_values_to_all_values(x):
        """
        Returns the percentage of values that are present in the time series
        more than once.
    
            len(different values occurring more than once) / len(different values)
    
        This means the percentage is normalized to the number of unique values,
        in contrast to the percentage_of_reoccurring_datapoints_to_all_datapoints.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if len(x) == 0:
            return np.nan
    
        unique, counts = np.unique(x, return_counts=True)
    
        if counts.shape[0] == 0:
            return 0
    
        return np.sum(counts > 1) / float(counts.shape[0])
        
    # 18)
    
    def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
        """
        Returns the percentage of non-unique data points. Non-unique means that they are
        contained another time in the time series again.
    
            # of data points occurring more than once / # of all data points
    
        This means the ratio is normalized to the number of data points in the time series,
        in contrast to the percentage_of_reoccurring_values_to_all_values.
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :return: the value of this feature
        :return type: float
        """
        if len(x) == 0:
            return np.nan
    
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
    
        value_counts = x.value_counts()
        reoccuring_values = value_counts[value_counts > 1].sum()
    
        if np.isnan(reoccuring_values):
            return 0
    
        return reoccuring_values / x.size
        
    # 19) 
    
    def sample_entropy(x):
        """
        Calculate and return sample entropy of x.
    
        .. rubric:: References
    
        |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
        |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
    
        :return: the value of this feature
        :return type: float
        """
        x = np.array(x)
    
        # if one of the values is NaN, we can not compute anything meaningful
        if np.isnan(x).any():
            return np.nan
    
        m = 2  # common value for m, according to wikipedia...
        tolerance = 0.2 * np.std(
            x
        )  # 0.2 is a common value for r, according to wikipedia...
    
        # Split time series and save all templates of length m
        # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
        
        ##################################################################################################
        def _into_subchunks(x, subchunk_length, every_n=1):
              """
              Split the time series x into subwindows of length "subchunk_length", starting every "every_n".
        
              For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix
        
                0  2  4
                1  3  5
                2  4  6
        
              with the settings subchunk_length = 3 and every_n = 2
              """
              len_x = len(x)
        
              assert subchunk_length > 1
              assert every_n > 0
        
              # how often can we shift a window of size subchunk_length over the input?
              num_shifts = (len_x - subchunk_length) // every_n + 1
              shift_starts = every_n * np.arange(num_shifts)
              indices = np.arange(subchunk_length)
        
              indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
              return np.asarray(x)[indexer]
        ######################################################################################
    
        xm = _into_subchunks(x, m)
    
        # Now calculate the maximum distance between each of those pairs
        #   np.abs(xmi - xm).max(axis=1)
        # and check how many are below the tolerance.
        # For speed reasons, we are not doing this in a nested for loop,
        # but with numpy magic.
        # Example:
        # if x = [1, 2, 3]
        # then xm = [[1, 2], [2, 3]]
        # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
        # and from [2, 3] => [[1, 1], [0, 0]]
        # taking the abs and max gives us:
        # [0, 1] and [1, 0]
        # as the diagonal elements are always 0, we substract 1.
        B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])
    
        # Similar for computing A
        xmp1 = _into_subchunks(x, m + 1)
    
        A = np.sum(
            [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
        )
    
        # Return SampEn
        return -np.log(A / B)
    
    
    ############################################################################################################
    # Extraction
    ############################################################################################################
    print("extraction")
    def feature_extraction(data):
        print(data.columns)  
        data_extraction = pd.DataFrame()
        print('to dentro')  
        for i in data.columns:
            vet = np.array(data['{}'.format(i)])
            l_var = []
            l_var.append(sum_values(vet))
            l_var.append(median(vet))
            l_var.append(mean(vet))
            l_var.append(length(vet))
            l_var.append(standard_deviation(vet))
            l_var.append(variance(vet))
            l_var.append(root_mean_square(vet))
            l_var.append(maximum(vet))
            l_var.append(minimum(vet))
            l_var.append(abs_energy(vet))
            l_var.append(mean_abs_change(vet))
            l_var.append(mean_change(vet))
            l_var.append(mean_second_derivative_central(vet))
            l_var.append(variation_coefficient(vet))
            l_var.append(skewness(vet))
            l_var.append(kurtosis(vet))
            l_var.append(percentage_of_reoccurring_values_to_all_values(vet))
            l_var.append(percentage_of_reoccurring_datapoints_to_all_datapoints(vet))
            l_var.append(sample_entropy(vet))
        
            features = ['{}_sum_values'.format(i), '{}_median'.format(i), '{}_mean'.format(i), '{}_length'.format(i), '{}_standard_deviation'.format(i), 
                          '{}_variance'.format(i), '{}_root_mean_square'.format(i), '{}_maximum'.format(i), '{}_minimum'.format(i), '{}_abs_energy'.format(i), 
                          '{}_mean_abs_change'.format(i), '{}_mean_change'.format(i), '{}_mean_second_derivative_central'.format(i), 
                          '{}_variation_coefficient'.format(i), '{}_skewness'.format(i), '{}_kurtosis'.format(i), '{}_percentage_of_reoccurring_values_to_all_values'.format(i), 
                          '{}_percentage_of_reoccurring_datapoints_to_all_datapoints'.format(i), '{}_sample_entropy'.format(i)]
        
            new_row = {'{}_sum_values'.format(i):l_var[0], '{}_median'.format(i):l_var[1], '{}_mean'.format(i):l_var[2], '{}_length'.format(i):l_var[3], 
                       '{}_standard_deviation'.format(i):l_var[4], '{}_variance'.format(i):l_var[5], '{}_root_mean_square'.format(i):l_var[6], '{}_maximum'.format(i):l_var[7], '{}_minimum'.format(i):l_var[8], 
                       '{}_abs_energy'.format(i):l_var[9], '{}_mean_abs_change'.format(i):l_var[10], '{}_mean_change'.format(i):l_var[11], '{}_mean_second_derivative_central'.format(i):l_var[12], 
                       '{}_variation_coefficient'.format(i):l_var[13], '{}_skewness'.format(i):l_var[14], '{}_kurtosis'.format(i):l_var[15], '{}_percentage_of_reoccurring_values_to_all_values'.format(i):l_var[16], 
                       '{}_percentage_of_reoccurring_datapoints_to_all_datapoints'.format(i):l_var[17], '{}_sample_entropy'.format(i):l_var[18]}
          
            df_aux = pd.DataFrame(columns=features)
            df_aux = df_aux.append(new_row, ignore_index=True)
            data_extraction = data_extraction.merge(df_aux, right_index=True, left_index=True, how='outer' )
            data_extraction = data_extraction.fillna(0)
            #print(l_var) 
            #print(features)
            #print(data_extraction.columns)   
        print(data_extraction) 
        return data_extraction    
    
    ###############################################################################################################################################
    # INVOKING
    ###############################################################################################################################################
    print('yum')  
    extraction = feature_extraction(data_n)
    
    print('tem') 
    print(type(extraction))
    
    # ********************
    # ******************************************
    s_3 = boto3.client('s3', aws_access_key_id='********************', aws_secret_access_key='******************************************' )
    
    name = json_file_name + '.csv'
    from io import StringIO 
    csv_buf = StringIO()
    extraction.to_csv(csv_buf, header=True, index=False)
    csv_buf.seek(0)
    s_3.put_object(Bucket='3wfeatures', Body=csv_buf.getvalue(), Key=name)  
    
    return 0    