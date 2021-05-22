# =============================================================================
# ELEXON API from DA Margin and Imbalance Data
#
# MELNGC - Indicated Margin/Maximum Export Limit
# IMBALNGC - Indicated Imbalance
# =============================================================================

import httplib2
import re
import pandas as pd


def post_elexon(url):

    # request and extract only content
    http_obj = httplib2.Http()
    content = http_obj.request(
        uri=url,
        method="GET",
        headers={"Content-Type": "application/xml; charset=UTF-8"},
    )[1]

    # strip and divide values
    stripped = re.sub("<[^<]+?>", " ", content.decode())

    # lists with each element data from each SP
    data = stripped.split()[16:]
    print(len(data))
    data = [data[x : x + 8] for x in range(0, len(data), 8)]
    print(len(data))

    return data


# PUT DATA IN DATAFRAME

# create empy lists for useful data
index = []
MELNGC = []
IMBALNGC = []

days_16 = ["31", "29", "31", "30", "31", "30", "31", "31", "30", "31", "30", "31"]
days = ["31", "28", "31", "30", "31", "30", "31", "31", "30", "31", "30", "31"]
month = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
years = ["2016", "2017", "2018"]

for i in years:
    if i == "2016":
        for j in range(0, len(month)):
            data = post_elexon(
                "https://api.bmreports.com/BMRS/MELIMBALNGC/v1?APIKey=b297kkm9vfw946g&ZoneIdentifier=N&FromDate={}-{}-01&ToDate={}-{}-{}&ServiceType=xml".format(
                    i, month[j], i, month[j], days_16[j]
                )
            )
            # atach useful data
            print(i, month[j], i, days_16[j])
            for k in range(0, len(data)):
                if data[k][0] == "DAM":
                    MELNGC.append(data[k][-2])
                    a = data[k][1].replace("-", "") + data[k][2]
                    index.append(a)
                if data[k][0] == "DAI":
                    IMBALNGC.append(data[k][-2])
    else:
        for j in range(0, len(month)):
            data = post_elexon(
                "https://api.bmreports.com/BMRS/MELIMBALNGC/v1?APIKey=b297kkm9vfw946g&ZoneIdentifier=N&FromDate={}-{}-01&ToDate={}-{}-{}&ServiceType=xml".format(
                    i, month[j], i, month[j], days[j]
                )
            )
            # atach useful data
            print(i, month[j], i, days[j])
            for k in range(0, len(data)):
                if data[k][0] == "DAM":
                    MELNGC.append(data[k][-2])
                    a = data[k][1].replace("-", "") + data[k][2]
                    index.append(a)
                if data[k][0] == "DAI":
                    IMBALNGC.append(data[k][-2])

DA_margin_imb = pd.DataFrame({"index": index, "DA_margin": MELNGC, "DA_imb": IMBALNGC})

DA_margin_imb.to_csv("margin_imbalance.csv")
