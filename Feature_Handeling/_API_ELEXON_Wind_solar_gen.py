# =============================================================================
# ELEXON API for Wind and Solar DATA FORECAST
# solar = Solar
# wind_on = Wind onshore
# wind_off = Wind offshore
# =============================================================================

import httplib2
import re
import pandas as pd

from numba import jit

solar = []
wind_on = []
wind_off = []

years = ['2016', '2017', '2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

@jit
def post_elexon(url):
    
    # request and extract only content
    http_obj = httplib2.Http()
    content = http_obj.request(
    uri=url,
    method='GET',
    headers={'Content-Type': 'application/xml; charset=UTF-8'},
    )[1]
     
    # strip and divide values
    stripped = re.sub('<[^<]+?>', ' ', content.decode())
    
    return stripped.split()

dates = []

for y in years:
    if y == '2016':
        for m in months:
            if m == '02':
                for d in range(1, 30):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        
            if (m == '01' or m == '03' or m == '05' or m == '07' or m == '08' or m == '10' or m == '12'):
                for d in range(1, 32):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        
            else:
                for d in range(1, 31):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        
                        
    else:
        pass
        '''
        for m in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            if m == '02':
                for d in range(1, 29):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        
            if (m == '01' or m == '03' or m == '05' or m == '07' or m == '08' or m == '10' or m == '12'):
                for d in range(1, 32):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        
            else:
                for d in range(1, 31):
                    for sp in range(1, 49):
                        dates.append([y,m,str(d).zfill(2),str(sp)])
                        data = post_elexon('https://api.bmreports.com/BMRS/B1440/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&Period={}&ServiceType=xml'.format(y,m,str(d).zfill(2),str(sp)))
                        look = []
                        for j in range(0, 72):
                            if data[j].lower() == 'wind':
                                look.append(data[j-1])                        
                        solar.append(look[0])
                        wind_off.append(look[2])
                        wind_on.append(look[4])
                        '''
    
