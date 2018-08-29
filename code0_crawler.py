#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 22:27:15 2017

@author: laihongji
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import pandas as pd
import random

#尝试设置ip代理，，失败:(
def get_ip_list(url, headers):
    r = requests.get(url, headers=headers).text
    soup = BeautifulSoup(r, 'lxml')
    ips = soup.find_all('tr')
    ip_list = []
    for i in range(1, len(ips)):
        ip_info = ips[i]
        tds = ip_info.find_all('td')
        ip_list.append(tds[1].text + ':' + tds[2].text)
    return ip_list

def get_random_ip(ip_list):
    proxy_list = []
    for ip in ip_list:
        proxy_list.append('http://' + ip)
    proxy_ip = random.choice(proxy_list)
    proxies = {'http': proxy_ip}
    return proxies
    
ip_list = get_ip_list('http://www.xicidaili.com/wn/', headers)

def ip_filter(ip_list, k):
    ip_list_new = []
    url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv46328&productId=3245078&score=' + str(k) + '&sortType=5&page=' + str(100) + '&pageSize=10&isShadowSku=0&rid=0&fold=1'
    for proxy in ip_list:
        try:
            p = {'http': proxy, 'https': proxy}
            res = requests.get(url, headers = headers, proxies=p, timeout = 1).text
            if res:
                #print('%s is OK' % proxy)
                #print('\n')
                ip_list_new_1.append(p)
        except Exception as e:
            #print('%s is FAILED' % proxy)
            #print(e)
            continue
    return ip_list_new

ip_list_new_1 = ip_filter(ip_list, 1)
ip_list_new_2 = ip_filter(ip_list, 2)
ip_list_new_3 = ip_filter(ip_list, 3)
ip_list_new_4 = ip_filter(ip_list, 4)
ip_list_new_5 = ip_filter(ip_list, 5)

#京东爬虫
scores = []
comments = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
for k in [1,2,3,4,5]:
    print('第%s级评论开始爬取...' % str(k))
    i = 0
    url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv46328&productId=3245078&score=' + str(k) + '&sortType=5&page=' + str(i) + '&pageSize=10&isShadowSku=0&rid=0&fold=1'
    r = requests.get(url, headers = headers).text
    while r:
        t = json.loads(r[27:len(r)-2])
        if t['comments']:
            for j in t['comments']:
                scores.append(j['score'])
                if 'afterUserComment' in j.keys():
                    comments.append(j['afterUserComment']['hAfterUserComment']['content'] + j['content'])
                else:
                    comments.append(j['content'])
        else:
            print('没有评论，停止抓取，总共爬取%s页' % str(i+1))
            break
        if (i+1) % 100 == 0:
            print('第%s页爬取完毕...' % str(i+1))
        if (i+1) % 20 == 0:
            time.sleep(15)
        i = i + 1
        url = 'https://sclub.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98vv46328&productId=3245078&score=' + str(k) + '&sortType=5&page=' + str(i) + '&pageSize=10&isShadowSku=0&rid=0&fold=1'
        r = requests.get(url, headers = headers).text
    else:
        print('没有评论，停止抓取，总共爬取%s页' % str(i+1))
    time.sleep(500)

data_jd = pd.DataFrame({"Comment":comments, "Score":scores})
data_jd.to_csv()

#由于设置ip失败，会被京东封ip，因此后面手动爬取了一部分数据。

#华为爬虫
i = 0
url = 'https://remark.vmall.com/remark/queryEvaluate.json?pid=938665621&pageNumber=' + str(i) +  '&callback=jsonp1514258590286'
r = requests.get(url).text
r2 = requests.get(url).text
comments = []
scores = []

while r:
    t = json.loads(r2[19:len(r2)-1])
    if t['remarkList']:
        for j in t['remarkList']:
            comments.append(j['content'])
            scores.append(j['score'])
    else:
        print('第%s页爬取完毕...' % str(i+1))
        break
    i = i + 1
    url = 'https://remark.vmall.com/remark/queryEvaluate.json?pid=938665621&pageNumber=' + str(i) +  '&callback=jsonp1514258590286'
    r = requests.get(url).text
    r2 = requests.get(url).text
    if i >= 10000:
        break
    if (i+1) % 100 == 0:
        print('第%s页爬取完毕...' % str(i+1))
    if (i+1) % 1000 == 0:
        time.sleep(10)