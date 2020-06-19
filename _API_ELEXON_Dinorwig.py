# =============================================================================
# API ELEXON
#
# From Detailed System Prices
#
# Binary feature if offers have been accepted by Dinorwig plant
# =============================================================================

import httplib2
import re
import pandas as pd
from numba import jit


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
    data = stripped.split()[34:]
    data = [data[x:x+24] for x in range(0, len(data),24)]
    
    return data

data = post_elexon('https://api.bmreports.com/BMRS/DETSYSPRICES/v1?APIKey=b297kkm9vfw946g&SettlementDate=2018-01-01&SettlementPeriod=3&ServiceType=xml') 

dino_bin = pd.read_csv('Dino.csv', usecols = [1]) 
dino_bin.columns = ['dino']
dino_bin = dino_bin['dino'].to_list()   

a = pd.read_csv('.UK_DA_Margin_Imb_forecast.csv', usecols = [1])
index = a['index'].to_list()

for i in index[47103:]:
    
    y = str(i)[0:4]
    m = str(i)[4:6]
    d = str(i)[6:8]
    sp = str(i)[8:]
    data = post_elexon('https://api.bmreports.com/BMRS/DETSYSPRICES/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&SettlementPeriod={}&ServiceType=xml'.format(y, m, d, sp))
    c = 0

    for i in data:
        if i[0] == 'OFFER':
            if (i[4] == 'T_DINO-1' or i[4] == 'T_DINO-2' or i[4] == 'T_DINO-3' or i[4] == 'T_DINO-4' or i[4] == 'T_DINO-5' or i[4] == 'T_DINO-6'):
                c = 1
            else:
                c = c
        else:
            c = c
            
    dino_bin.append(c)

# create data frame and save as csv
dino_bin = pd.DataFrame(dino_bin, index = index)
dino_bin.columns = ['dino_bin']
dino_bin.to_csv(r'C:\Users\maria\OneDrive - Imperial College London\SEF-DESKTOP-72DBAPV\THESIS\Python_Coding\.UK_Dinorwig_presence.csv')
    