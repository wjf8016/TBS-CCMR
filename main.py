from Rec import Rec
from util.conf import ModelConf

if __name__ == '__main__':
    baseline = []
    graph_models = []
    sequential_models = []
    multi_behavior_models = ['TBSCCMR']

    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print('=' * 80)

    print('Baseline Models:')
    print('   '.join(baseline))
    print('-' * 80)
    print('Graph-Based Models:')
    print('   '.join(graph_models))

    print('-' * 80)
    print('Multi-Behavior Models:')
    print('   '.join(multi_behavior_models))

    print('=' * 80)
    model = 'TBSCCMR'
    import time

    s = time.time()
    if model in baseline or model in graph_models or model in sequential_models or model in multi_behavior_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = Rec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
