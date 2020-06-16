# =============================================================================
# ELEXON API -> CALLING ELEXON RESTful API
# =============================================================================

import httplib2
import re

def post_elexon(url):
 http_obj = httplib2.Http()
 content = http_obj.request(
 uri=url,
 method='GET',
 headers={'Content-Type': 'application/xml; charset=UTF-8'},
 )[1]
 #print(content.decode())
 stripped = re.sub('<[^<]+?>', ' ', content.decode())
 print(stripped.split())
 print(len(stripped.split()))
 print(stripped.split()[14])
 #print(resp.status)

def main():
 post_elexon(
 url='https://api.bmreports.com/BMRS/B1770/v1?APIKey=b297kkm9vfw946g&SettlementDate=2015-03-01&Period=1&ServiceType=xml',
 ) 
 
if __name__ == "__main__":
     main() 
     
# PUT DATA IN DATAFRAME

