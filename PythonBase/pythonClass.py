class t1(object):
    def __init__(self):
        print "aaa"
    def my(self):
        print "ccc"
    def __del__(self):
        print "del success"

obj=t1()
obj.my()
del obj


#obj.my() NameError: name 'obj' is not defined