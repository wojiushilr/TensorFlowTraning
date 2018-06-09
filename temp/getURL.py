#coding=utf-8
import urllib

def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html

html = getHtml("http://pubchemqc.riken.jp:35000/view_gamess_td_log/68240?nsukey=bOyHfhp7im4DyCNkVRLg4wik1gis%2FXQlIjSRxX2UVpAUhS5bbFlzkQh%2Bps0KFiACOsOAZl3Rc08ZqytDH6a443ITsqQSQV22XSYaUBEyDUonkyzR14wLk9538cCKQLyXt%2BENc0OkE%2BJf07oxyNguUS37VLeLdqtIOQaS60v%2B8N7AUaw1rGXScd%2BvB%2FKBlYq3")

print html