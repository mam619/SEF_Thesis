# =============================================================================
# Peak Wind Generation Forecast
#
# Time of peak [2]
# Peak (Max) MW [3]
# Total Metered Capacity (MW) [4]
# =============================================================================
import httplib2
import re
import pandas as pd

from numba import jit

index = []
time = []
peak = []
total_cap = []

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
    
    return stripped.split()[11:]

# first 5 months
data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2016-01-01&ToDate=2016-05-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])

# second 5 months   
data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2016-05-02&ToDate=2016-09-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])
    
data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2016-09-02&ToDate=2017-01-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])

data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2017-01-02&ToDate=2017-05-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])

data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2017-05-02&ToDate=2017-09-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])

data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2017-09-02&ToDate=2018-01-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])

data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2018-01-02&ToDate=2018-05-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])
    
data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2018-05-02&ToDate=2018-09-01&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])
    
data = post_elexon('https://api.bmreports.com/BMRS/WINDFORPK/v1?APIKey=b297kkm9vfw946g&FromDate=2018-09-02&ToDate=2018-12-31&ServiceType=xml')

data = [data[x:x+8] for x in range(0, len(data),8)]

for i in range(0, len(data)):
    index.append(data[i][1].replace('-', ''))
    time.append(data[i][2].replace(':', ''))
    peak.append(data[i][3])
    total_cap.append(data[i][4])


# create a data frame    
wind_peak = pd.DataFrame({'index': index, 'time_peak': time, 'peak(MW)': peak, 'total_cap(MW)': total_cap})

# save as csv
wind_peak.to_csv(r'C:\Users\maria\Documents\Wind_peaks_daily_FORECAST.csv')