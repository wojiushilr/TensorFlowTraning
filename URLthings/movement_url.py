# -*- coding: utf-8 -*-
# 引入selenium中的webdriver
# 针对自动跳转对网页爬取
from selenium import webdriver
import time

# webdriver中的PhantomJS方法可以打开一个我们下载的静默浏览器。
# 输入executable_path为当前文件夹下的phantomjs.exe以启动浏览器
driver = webdriver.PhantomJS("/usr/local/bin/phantomjs")

# 使用浏览器请求页面
driver.get("http://pubchemqc.riken.jp:35000/view_gamess_td_log/68240")
# 加载3秒，等待所有数据加载完毕
time.sleep(3)

# 通过id来定位元素，
# .text获取元素的文本数据
print driver.find_element_by_id('gamess_file').text

# 关闭浏览器
driver.close()