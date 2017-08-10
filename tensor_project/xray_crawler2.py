from lxml import html
import requests
import re
import json
import urllib
import sys
import csv
import os

path = 'D:\\data\\python\\x-ray\\image\\'
label_path = 'D:\\data\\python\\x-ray\\'

domain = 'https://openi.nlm.nih.gov/'
url_list = []
for i in range(0,75):
    url = 'https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m='+str(1+100*i)+'&n='+str(100+100*i)
    url_list.append(url)
regex = re.compile(r"var oi = (.*);")
final_data = {}
img_no = 0


def extract(url):
    global img_no

    img_no += 1
    r = requests.get(url)
    tree = html.fromstring(r.text)

    div = tree.xpath('//table[@class="masterresultstable"]\
        //div[@class="meshtext-wrapper-left"]')

    if div != []:
        div = div[0]
    else:
        return

    typ = div.xpath('.//strong/text()')[0]
    items = div.xpath('.//li/text()')[0]
    img = tree.xpath('//img[@id="theImage"]/@src')[0]


    final_data[img_no] = {}
    final_data[img_no]['type'] = typ
    final_data[img_no]['items'] = items
    final_data[img_no]['img'] = domain + img

    if not os.path.exists(path+str(img_no)+".png"):
        urllib.request.urlretrieve(domain+img, path+str(img_no)+".png")
        with open('data_new.json', 'w') as f:
            json.dump(final_data, f)

    with open(label_path + 'xray_labels.csv', 'a', encoding='utf-8') as f:
        if items == "normal":
            f.write(items + '\n')
        else:
            f.write('patient\n')
    print(img_no)


def main():
    for url in url_list :
        global img_no

        r = requests.get(url)
        tree = html.fromstring(r.text)

        script = tree.xpath('//script[@language="javascript"]/text()')[0]

        json_string = regex.findall(script)[0]
        json_data = json.loads(json_string)

        next_page_url = tree.xpath('//footer/a/@href')

        print('extract')
        links = [domain + x['nodeRef'] for x in json_data]

        for link in links:
            extract(link)

if __name__ == '__main__':

    main()


#python scraper.py <path to folders>