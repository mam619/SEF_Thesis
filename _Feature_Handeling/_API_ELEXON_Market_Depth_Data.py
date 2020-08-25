# =============================================================================
# Market Depth DATA
# 
# Offer Volume (MWh) [4]
# Bid Volume (MWh) [5]
# Accepted Offer Vol (MWh) [6]
# Accepted Bid Vol (MWh) [7]
# =============================================================================

import httplib2
import re
import pandas as pd
import numpy as np

from numba import jit

for_index = []
offer_vol = []
bid_vol = []
accepted_offer_vol = []
accepted_bid_vol = []


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
    data = stripped.split()[10:]
    data = [data[x:x+13] for x in range(0, len(data),13)]
    return data

a = pd.read_csv('DA_Margin_Imb_FORECAST.csv', usecols = [1])
index = a['index'].to_list()
index = [str(index[x])[:8] for x in range(0, len(index))]
index = np.unique(index)

for i in index:
    y = i[0:4]
    m = i[4:6]
    d = i[6:8]
    data = post_elexon('https://api.bmreports.com/BMRS/MKTDEPTHDATA/v1?APIKey=b297kkm9vfw946g&SettlementDate={}-{}-{}&ServiceType=xml'.format(y, m, d))
    for j in range(0, len(data)):
        for_index.append(i + data[j][2])
        offer_vol.append(data[j][4])
        bid_vol.append(data[j][5])
        accepted_offer_vol.append(data[j][6])
        accepted_bid_vol.append(data[j][7])
        
Market_Depth_Data = pd.DataFrame({'index': for_index , 'Offer_vol': offer_vol, 'Bid_vol': bid_vol, 'Accepted_offer_vol': accepted_offer_vol, 'Accepted_bid_vol': accepted_bid_vol})

Market_Depth_Data.to_csv('Market_depth_data.csv')