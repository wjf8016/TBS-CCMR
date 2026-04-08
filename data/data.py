class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training[:]
        self.test_data = test[:]

class MBData(object):
    def __init__(self, conf, training_p, test, training_c, training_v,type_num):
        self.config = conf
        self.training_p = training_p[:]
        self.training_c = training_c[:]
        self.training_v = training_v[:]
        self.test_data = test[:]
        self.type_num = type_num





