#import xml.etree.ElementTree as ET
#import sys
#
#reload(sys)
#sys.setdefaultencoding( "UTF-8" )
#tree = ET.parse('reut2-001.sgm')
#root = tree.getroot()
#
#print(root.tag)
#


import BeautifulSoup
from BeautifulSoup import BeautifulSoup

with open('reut2-001.sgm', 'r') as myfile:
    data=myfile.read().replace('\n', '')

print(len(data))
soup = BeautifulSoup(data)
docs = soup.findAll('body')

for doc in docs:
    print(doc.text)

