import math
import scipy.stats

class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num/total_num

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return error/count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return math.sqrt(error/count)

    @staticmethod
    def NDCG(origin,res,N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            #1 = related, 0 = unrelated
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG+= 1.0/math.log(n+2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG+=1.0/math.log(n+2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)

    @staticmethod
    def Nov(origin,res,N,data):
        user_num = data.user_num
        degree_dict = data.degree
        # matrix = data.ui_adj
        Nov_score = 0
        #先遍历所有user，再遍历topk item
        for i, user in enumerate(origin):
            for item, score in res[user][:N]:
                #当交互次数大于1时的计算方法
                degree = degree_dict[item]
                if degree == 0:
                    tem_socre = 0
                else:
                    tem_socre = -(1/math.log2(user_num))*(math.log2(degree/user_num))
                Nov_score = Nov_score + tem_socre
        Nov_score = Nov_score * (1/(user_num*N))

        return Nov_score


    @staticmethod
    def PRU(origin,res,N,data):
        #res_list 应该是降序的
        user_num = data.user_num
        degree_dict = data.degree
        # matrix = data.ui_adj
        PRU_score = 0
        for i, user in enumerate(origin):
            degree_list = []
            rank_list = [i for i in range(1, N + 1)]
            for item, score in res[user][:N]:
                # 当交互次数都是1时的计算方法
                # degree = len(matrix[user_num + data.item[item], :].nonzero()[1])
                # 当交互次数大于1时的计算方法
                degree = degree_dict[item]
                degree_list.append(degree)
                #不确定是correlation还是pvalue，0是correlation，1是pvalue
            PRU_score = PRU_score + scipy.stats.spearmanr(degree_list, rank_list)[0]
        PRU_score = PRU_score * (-1/user_num)

        return PRU_score


def ranking_evaluation(origin, res, N):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def ranking_evaluation1(origin, res, N, data):
    measure = []
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        Nov = Metric.Nov(origin, predicted, n, data)
        indicators.append('Nov:' + str(Nov) + '\n')
        PRU = Metric.PRU(origin, predicted, n, data)
        indicators.append('PRU:' + str(PRU) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure.append('Top ' + str(n) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure