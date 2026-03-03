import os
import importlib.util
import sys
import argparse
import cdsapi
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import time
import random
import  pandas as pd
import tempfile
import pdb
import copy
import zipfile
from cra5.utils.config import Config
from pandas._typing import  DatetimeLikeScalar

class era5_downloader():
    def __init__(self, config):
        self.cfg = Config.fromfile(config)
        self.proxies = self.env_seting()
        self.ecmwf_dataset_pressure = 'reanalysis-era5-pressure-levels'
        self.ecmwf_dataset_single = 'reanalysis-era5-single-levels'

        self.local_root = self.cfg.storage.local

        self.cdsapi_client = cdsapi.Client(url=os.environ.get("CDSAPI_URL"),
                                            key=os.environ.get("CDSAPI_KEY"))
        self.cdsapi_client.session.proxies.update(self.proxies)
        self.pressure_request_dic=copy.deepcopy(self.cfg.pressure_request_dic)
        self.single_request_dic=copy.deepcopy(self.cfg.single_request_dic)    
        
    def save(self, time_required, file_key):
        prefix, extension=os.path.splitext(file_key)

        request_dic =copy.deepcopy(self.pressure_request_dic)
        request_dic.update(time_required)
         
        file_name = f'{prefix}_pressure{extension}'
        print(f'File is saved to {file_name}')
        
        if self.check_filesize(file_name, request_dic, self.ecmwf_dataset_pressure, file_name) is False:
            print("{} does not been completely downloaded, "
                  "I would like to download {} again !!!".format(file_name, file_name))
        
            dir = os.path.dirname(file_name)
            if os.path.exists(dir) is not  True:
                os.makedirs(dir)

            self.cdsapi_client.retrieve(self.ecmwf_dataset_pressure, request_dic, file_name)
             
            self.save(time_required, file_key)
        else:
            print("{} has been fully downloaded, No need to down it again !!!".format(file_key))

        
        request_dic =copy.deepcopy(self.single_request_dic)
        request_dic.update(time_required)        
        file_name = f'{prefix}_single{extension}'
        print(f'File is saved to {file_name}') 
        
        if self.check_filesize(file_name, request_dic, self.ecmwf_dataset_single, file_name) is False:
            print("{} does not been completely downloaded, "
                  "I would like to download it again !!!".format(file_key))
            dir = os.path.dirname(file_name)
            if os.path.exists(dir) is not  True:
                os.makedirs(dir)
            
            # Download file (might be zip or nc)
            downloaded_file = file_name
            self.cdsapi_client.retrieve(self.ecmwf_dataset_single, request_dic, downloaded_file)
            
            # Check if downloaded file is a zip file
            if zipfile.is_zipfile(downloaded_file):
                print(f"Downloaded file is a zip file, extracting...")
                zip_path = downloaded_file
                extract_dir = os.path.dirname(zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract all files
                    zip_ref.extractall(extract_dir)
                    # Find the .nc file in the extracted files
                    extracted_files = zip_ref.namelist()
                    nc_files = [f for f in extracted_files if f.endswith('.nc')]
                    
                    if nc_files:
                        # Move the extracted .nc file to the expected location
                        extracted_nc = os.path.join(extract_dir, nc_files[1])
                        if extracted_nc != file_name:
                            if os.path.exists(file_name):
                                os.remove(file_name)
                            os.rename(extracted_nc, file_name)
                            print(f"Extracted {nc_files[1]} to {file_name}")
                        # Remove the zip file
                        print(f"Removed zip file: {zip_path}")
                    else:
                        print(f"Warning: No .nc file found in zip archive {zip_path}")
            
            self.save(time_required, file_key)
        else:
            print("{} has been fully downloaded, No need to down it again !!!".format(file_key))
            return True

    def check_filesize(self, file_key,request_dic, ecmwf_dataset, file_path):
        exist_size=0

        if not os.path.exists(file_path):
            exist_size = 0
        else:
            # file_path should always be .nc file (zip already extracted in save function)
            exist_size = os.path.getsize(file_path)
    
        # Get remote size
        remote_size = self.cdsapi_client.retrieve(ecmwf_dataset,
                                                    request_dic,).content_length
        
        # For single dataset, remote_size is the zip file size, but local file is 
        # the extracted first .nc file, so we can't directly compare sizes.
        # We check if the file exists and has reasonable size (> 0)
        if ecmwf_dataset == self.ecmwf_dataset_single:
            if exist_size > 0:
                # File exists and has content, consider it complete
                # (can't compare zip size with extracted file size)
                print("{} (extracted from zip) is complete, local size: {}, remote zip size: {}"
                      .format(file_key, exist_size, remote_size))
                return True
            else:
                print("{} is not complete or missing, local size: {}, remote zip size: {}"
                      .format(file_key, exist_size, remote_size))
                return False
        else:
            # For pressure level dataset, normal comparison (both are .nc files)
            if remote_size == exist_size:
                print("{} is complete, remote vs local size: {}=={}"
                      .format(file_key, remote_size, exist_size))
                return  True
            else:
                print("{} is not complete, remote vs local size: {}!={}"
                      .format(file_key, remote_size, exist_size))
                return False


    def env_seting(self):
        os.environ['CDSAPI_URL'] = 'https://cds.climate.copernicus.eu/api'
        os.environ['CDSAPI_KEY'] = 'ea3a2607-158c-48a4-bd27-b255256b2759'
        
        if self.cfg.proxy.type=='direct':
            proxies = {}
            print('you does not use any proxy !!!!')

        elif  self.cfg.proxy.type=='normal':
            proxies = dict(http=self.cfg.proxy.normal,
                 https=self.cfg.proxy.normal)

        elif self.cfg.proxy.type=='special':
            proxies = dict(http=self.cfg.proxy.special,
                                https=self.cfg.proxy.special)
            print("i am using a special proxy of %s" % self.cfg.proxy.special)

        else:
            raise  ValueError("proxy type must be 'direct', 'normal' or 'special' !!!! ")
        
        return proxies
    def get_yy_mm_dd_hh(self, time_stamp):
        time_stamp =  pd.to_datetime(time_stamp)
        print(time_stamp)
        yy = str(time_stamp.year).zfill(4)
        mm = str(time_stamp.month).zfill(2)
        dd = str(time_stamp.day).zfill(2)
        hh=str(time_stamp).split(' ')[-1]
        return yy, mm, dd, hh
    
    def get_form_timestamp(self,
                           time_stamp:str, 
                           local_root=None):
        yy, mm, dd, hh_mm_ss = self.get_yy_mm_dd_hh(time_stamp)
        time_required=dict(year = yy,month = mm, day = dd, time = hh_mm_ss)
        local_root = self.local_root or local_root
        file_path = f'{local_root}/{yy}/{time_stamp}.nc'
      
        self.save(time_required, file_path)
        
        
def formatSize(bytes,format='GB'):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("The bytes is not correct")
        return "Error"

    if format=="GB":
        return kb/1024/1024
    if format=="MB":
        return kb/1024