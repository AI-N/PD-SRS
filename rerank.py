import networkx as nx
import math
from math import comb
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#%matplotlib inline


parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float, default=0.5, help='alpha: to determine the proportion of long-tail items in top-k recommendation based on user LID')#[0.1, 0.5, 1.0]
opt,a= parser.parse_known_args()
print(opt)


## Re-ranking Mechanism
def LT_inclusion(scores_ind,alpha,df_t,lt,scores5000_ind):   #df_t is the dataframe for test sequences (refer to preprocess.py)
    alpha=opt.alpha  #hyper parameter to determine how many LT is needed based on the LID in current session
    LID=1-df_t['Sim_list']  #level of interest to diversity in current session s for user u
    D=alpha*LID
    scores_ind_new=[]
    for ix in range(len(scores_ind)):
        #print('how many LT items can be included based on its LID in current session:',math.floor(len(scores_ind[ix])*D[ix]))
        #print('scores_ind:',scores_ind[ix])
        #print(pd.Series(scores_ind[ix]).isin(lt))
        pos=[ii for ii, x in enumerate(pd.Series(scores_ind[ix]).isin(lt)) if x] #position (index) of long tail items in top-k recommendation
        #print('position of LT items in top-k list:',pos)
        if len(pos)>0:
            a=[]
            for j in range(len(pos)):
                a.append(scores_ind[ix][pos[j]])
            #print('\tLT items:',a)
        sum(pd.Series(scores_ind[ix]).isin(lt))  #count of long tail items in top-k recommendation
        #print('count of LT items in top-k list:',sum(pd.Series(scores_ind[ix]).isin(lt)))
        pos2=[ij for ij, x in enumerate(pd.Series(scores5000_ind[ix]).isin(lt)) if x]
        #print('position of LT items in top-500:',pos2)
        a=[]
        for p in range(len(pos2)):
            a.append(scores5000_ind[ix][pos2[p]])
        #print('\tLT items in top-500:',a)
        if sum(pd.Series(scores_ind[ix]).isin(lt)) < math.floor(len(scores_ind[ix])*D[ix]): #if the number of existed LT items is less than the number of needed
            to_add=[]
            for n in range(sum(pd.Series(scores_ind[ix]).isin(lt)),math.floor(len(scores_ind[ix])*D[ix])): #as much as needed (ignore the existed ones)
                to_add.append(a[n])
            #print(to_add)
            ar=np.arange(len(scores_ind[ix])).tolist()
            #print(ar)
            for element in pos:
                if element in ar:
                    ar.remove(element)
            ar.sort(reverse=True)
            #print('available indices to be replaced by LT items):',ar)
            new=scores_ind[ix]
            for c in range(len(to_add)):
                new[ar[c]]=to_add[c]
            scores_ind_new.append(new)
        else:
            scores_ind_new.append(scores_ind[ix])
    return scores_ind_new 

##evaluation 
def hit_mrr(scores_ind_new, targets_):
    hit,mrr=[],[]
    for score, target in zip(scores_ind_new, targets_):
        hit.append(np.isin(target - 1, score))
        if len(np.where(score == target - 1)[0]) == 0:
            mrr.append(0)
        else:
            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))  ##position of item in the list: the beter the topper it sits
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit,mrr


def diversity(scores_ind):
    div = []
    for i in range(len(scores_ind)):
        a = 0
        for j in range(len(scores_ind[0])):
            for k in range(len(scores_ind[0])):
                if j < k:
                    a += model_deepwalk.wv.similarity(scores_ind[i][j], scores_ind[i][k])
        div.append(a/comb(len(scores_ind[0]),2))  
    return sum(div)/len(div)

# 2D plot of items in the trained Word2Vev model
def plot_nodes(word_list):
    X = model_deepwalk.wv[word_list]
    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(X) 
    plt.figure(figsize=(14,14))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))     
    plt.show()

