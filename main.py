#main in pytorch

import argparse
import pickle
import time
import networkx as nx
import pandas as pd
from gensim.models import Word2Vec
import copy
from copy import deepcopy
from utils import *
from model_GNN import *
from rerank import *

torch.cuda.set_device(1)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Xing', help='dataset name: diginetica/Xing/Reddit')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--TA', default=False, help='use target-aware or not')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha: to determine the proportion of long-tail items in top-k recommendation based on user LID')#[0.1, 0.5, 1.0]
opt,a= parser.parse_known_args()
print(opt)

def main():
    train_data = pickle.load(open(opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'Xing':
        n_node = 59122
    elif opt.dataset == 'Reddit':
        n_node = 80000

    model = trans_to_cuda(SessionGraph(opt, n_node))
    #print(model)

    
    trainn=pd.read_csv(opt.dataset+'/trainn.csv', delimiter=',')
    testt=pd.read_csv(opt.dataset+'/testt.csv', delimiter=',')
    df_tr=pd.read_csv(opt.dataset+'/df_tr.csv', delimiter=',')
    df_t=pd.read_csv(opt.dataset+'/df_t.csv', delimiter=',')
    
    if opt.dataset == 'diginetica':
        dataset = 'diginetica/train-item-views.csv'
    elif opt.dataset =='Xing':
        dataset = 'Xing/xing.csv'
    elif opt.dataset =='Reddit':
        dataset = 'Reddit/reddit.csv'

    
    if opt.dataset == 'Xing' or opt.dataset == 'Reddit':      #Split out %20 of each user's sessions as test set 
        df = pd.read_csv(dataset, delimiter=',')
        df=df.drop(['Unnamed: 0'], axis=1)
    elif opt.dataset == 'diginetica':     #Split out test set based on dates (7 days for test)
        df = pd.read_csv(dataset, delimiter=';')
        df=df.drop(['userId'], axis=1)
        df=df.rename(columns={"sessionId": "session_id", "itemId": "item_id", "eventdate": "ts"})
        
    
    start = time.time()
    item_clicks=pd.read_csv(opt.dataset+'/item_clicks.csv', delimiter=',')
    G=nx.from_pandas_edgelist(item_clicks, "source", "target", edge_attr=None, create_using=nx.Graph())
    # function to generate random walk sequences of nodes
    def get_randomwalk(node, path_length):
        random_walk = [node]
        for i in range(path_length-1):
            temp = list(G.neighbors(node))
            temp = list(set(temp) - set(random_walk))    
            if len(temp) == 0:
                break
            random_node = random.choice(temp)
            random_walk.append(random_node)
            node = random_node  
        return random_walk 
    all_nodes = list(G.nodes())
    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(5):
            random_walks.append(get_randomwalk(n,10))

    # train word2vec model
    model_deepwalk = Word2Vec(window = 4, sg = 1, hs = 0,
                     negative = 10, # for negative sampling
                     alpha=0.03, min_alpha=0.0007,
                     seed = 14)
    print('DeepWalk model:',model_deepwalk)
    

    model_deepwalk.build_vocab(random_walks, progress_per=2)
    print('training...')
    model_deepwalk.train(random_walks, total_examples = model_deepwalk.corpus_count, epochs=20, report_delay=1)
    #terms=item_clicks['source'].unique()
    #plot_nodes(terms)
    
    print('DeepWalk model Done')
    print('Wait! it is computing the similarity of items in users current sessions')
    alpha=opt.alpha
    #final listwise sim : how similar are the items in the current session -> to predict top-k items as the next click
    df_t['Sim_list'] = ""   #df_t is the dataframe for test sequences (refer to preprocess.py) 
    for ind in range(len(df_t)):
        if len(df_t['test_seq'][ind].strip('][').split(', '))==1:
            df_t['Sim_list'][ind]=1
        else:
            tsim=[]
            for counter, i in enumerate(df_t['test_seq'][ind].strip('][').split(', ')[0:len(df_t['test_seq'][ind].strip('][').split(', '))-1]):
                for j in df_t['test_seq'][ind].strip('][').split(', ')[counter+1:len(df_t['test_seq'][ind].strip('][').split(', '))]:
                    sim=model_deepwalk.wv.similarity(int(i), int(j))   # find similarity between two items based on our Word2vec model
                    tsim.append(sim)
            df_t['Sim_list'][ind]=sum(tsim)/len(tsim)
    df_t.to_csv(opt.dataset+'/df_t.csv', header=True, index=False)
    end = time.time()
    print("Run time: %f s" % (end - start))
    print('Wait! it continues...')
    
    start2 = time.time()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit5, mrr5, hit10, mrr10, hit20, mrr20, targets_, scores5_value, scores5_ind, scores10_value, scores10_ind, scores20_value, scores20_ind, scores5000_value, scores5000_ind = train_test(model, train_data, test_data)
        flag = 0
        if hit5 >= best_result[0]:
            best_result[0] = hit5
            best_epoch[0] = epoch
            flag = 1
        if mrr5 >= best_result[1]:
            best_result[1] = mrr5
            best_epoch[1] = epoch
        if hit10 >= best_result[2]:
            best_result[2] = hit10
            best_epoch[2] = epoch
            flag = 1
        if mrr10 >= best_result[3]:
            best_result[3] = mrr10
            best_epoch[3] = epoch
        if hit20 >= best_result[4]:
            best_result[4] = hit20
            best_epoch[4] = epoch
            flag = 1
        if mrr20 >= best_result[5]:
            best_result[5] = mrr20
            best_epoch[5] = epoch
            flag = 1
        print('Best Result:')
        print('Recall@5:%.4f\tMMR@5:%.4f\tRecall@10:%.4f\tMMR@10:%.4f\tRecall@20:%.4f\tMMR@20:%.4f\tEpoch:%d,\t%d,\t%d,\t%d,\t%d,\t%d'% (best_result[0], best_result[1], 
                                                                                                                                         best_result[2], best_result[3], 
                                                                                                                                         best_result[4], best_result[5], 
                                                                                                                                         best_epoch[0], best_epoch[1], 
                                                                                                                                         best_epoch[2], best_epoch[3], 
                                                                                                                                         best_epoch[4], best_epoch[5]))
        
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('mean diversity of top-5:', diversity(scores5_ind),'mean diversity of top-10:',diversity(scores10_ind),'mean diversity of top-20:',diversity(scores20_ind))
    print('-------------------------------------------------------')
    end2 = time.time()
    print("Run time: %f s" % (end2 - start2))
    #print('targets[0:5]:', targets_[0:5], '\nscores20_ind[0:5]:', scores20_ind[0:5], '\nscores20_value[0:5]:', scores20_value[0:5])
    print('Wait! it continues...')
    

    start3 = time.time()
    ## unpopularity calculation
    pop=[]
    for i in df['item'].unique():
        pop.append(len(trainn[trainn['item']==i])/len(df['item'].unique()))
    pop_max=max(pop)
    pop_df=pd.DataFrame(pop, columns = ['pop'])
    pop=pop_df/pop_max
    unpop_=1-(pop_df/pop_max)
    unpop=unpop_.rename(columns={"pop": "unpop"})


    ## Assign set of long-tail items
    ind=unpop[unpop['unpop']>=0.95].index   #can be chosen to have 10% of less popular items as long-tail items
    #print(ind)
    lt=df['item'][ind].tolist()
    #print('These two should have same length (if not: something went wrong):',len(LID),len(scores5_ind))
    alpha=opt.alpha
    scores5_ind1 = deepcopy(scores5_ind)
    scores5_ind_new=LT_inclusion(scores5_ind,alpha,df_t,lt,scores5000_ind)

    #top-5:   and top-10 and top-20 can also be calculated this way...
    be=list(set([item for sublist in scores5_ind1 for item in sublist]))
    af=list(set([item for sublist in scores5_ind_new for item in sublist]))
    print('\nnum of LT items (before):',sum(pd.Series(be).isin(lt)),'\nnum of LT items (the proposed method):',sum(pd.Series(af).isin(lt)),
          '\ntotal number of LT items:',len(lt))
    print('\nLT coverage (before):',sum(pd.Series(be).isin(lt))/len(lt),'\nLT coverage (the proposed method):',sum(pd.Series(af).isin(lt))/len(lt),
          '\nimprovement (times):',sum(pd.Series(af).isin(lt))/sum(pd.Series(be).isin(lt)))
   
    print('Recall@5 and MRR@5:',hit_mrr(scores5_ind_new, targets_))
    print('mean diversity of top-5:', diversity(scores5_ind_new))

    print('-------------------------------------------------------')
    end3 = time.time()
    print("Run time: %f s" % (end3 - start3))
    
if __name__ == '__main__':
    main()
