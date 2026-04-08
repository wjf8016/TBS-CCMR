from data.loader import FileIO


class Rec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        print(config['model.type'])
        if config['model.type'] == 'sequential':
            self.training_data, self.test_data = FileIO.load_data_set(config['sequence.data'], config['model.type'])
        elif config['model.type'] == 'multiBehavior':
            self.training_p = FileIO.load_data_set(config['training_p.set'], config['model.type'])
            self.training_c = FileIO.load_data_set(config['training_c.set'], config['model.type'])
            self.training_v = FileIO.load_data_set(config['training_v.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])
            self.type_num = FileIO.load_pickle_file(config['type_num.set'])
        else:
            self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        print('Reading data and preprocessing...')

    def execute(self):
        import_str = 'from model.'+ self.config['model.type'] +'.' + self.config['model.name'] + ' import ' + self.config['model.name']
        print(import_str)
        exec(import_str)
        print(self.config['model.type'])
        if self.config['model.type'] == 'multiBehavior':
            recommender = self.config['model.name'] + '(self.config,self.training_p,self.test_data,self.training_c,self.training_v,self.type_num,**self.kwargs)'
        if self.config['model.type'] == 'Multi_behavior':
            recommender = self.config['model.name'] + '(self.config,self.training_p,self.test_data,self.training_f,self.training_c,self.training_v,self.type_num,**self.kwargs)'
        if self.config['model.type'] == 'graph':
            recommender = self.config['model.name'] + '(self.config,self.training_data,self.test_data,**self.kwargs)'
        eval(recommender).execute()
