#preprocess 

import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pandas as pd
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Xing', help='dataset name: diginetica/Xing/Reddit')
opt,a= parser.parse_known_args()
print(opt)


if opt.dataset == 'diginetica':
    dataset = 'diginetica/train-item-views.csv'
elif opt.dataset =='Xing':
    dataset = 'Xing/xing.csv'
elif opt.dataset =='Reddit':
    dataset = 'Reddit/reddit.csv'

print("-- Starting @ %ss" % datetime.datetime.now())
start = time.time()
with open(dataset, "r") as f:
    if opt.dataset == 'Xing' or opt.dataset == 'Reddit':
        reader = csv.DictReader(f, delimiter=',')
        sess_clicks = {}
        user_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        for data in reader:
            sessid = data['session_id']
            uid = data['user']
            curid = sessid
            item = data['user'], data['item'], int(data['ts'])
            user=data['session_id'],data['item'],int(data['ts'])
            if uid in user_clicks:
                user_clicks[uid] += [user]
            else:
                user_clicks[uid] = [user]

            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        print(sess_clicks['0'])
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[1] for c in sorted_clicks]
    
    elif opt.dataset == 'diginetica':
        reader = csv.DictReader(f, delimiter=';')
        sess_clicks = {}
        sess_date = {}
        ctr = 0
        curid = -1
        curdate = None
        for data in reader:
            sessid = data['sessionId']
            if curdate and not curid == sessid:
                date = ''
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                sess_date[curid] = date
            curid = sessid
            item = data['itemId'], int(data['timeframe'])
            curdate = ''
            curdate = data['eventdate']
            if sessid in sess_clicks:
                sess_clicks[sessid] += [item]
            else:
                sess_clicks[sessid] = [item]
            ctr += 1
        date = ''
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            sess_clicks[i] = [c[0] for c in sorted_clicks]
        sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())

## Filter out length 1 sessions
#for s in list(sess_clicks):
#    if len(sess_clicks[s]) == 1:
#        del sess_clicks[s]
#        if opt.dataset == 'diginetica':
#            del sess_date[s]

## Count number of times each item appears
#iid_counts = {}
#for s in sess_clicks:
#    seq = sess_clicks[s]
#    for iid in seq:
#        if iid in iid_counts:
#            iid_counts[iid] += 1
#        else:
#            iid_counts[iid] = 1

#sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

#length = len(sess_clicks)
#for s in list(sess_clicks):
#    curseq = sess_clicks[s]
#    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
#    if len(filseq) < 2:
#        del sess_clicks[s]
#        if opt.dataset == 'diginetica':
#            del sess_date[s]
#    else:
#        sess_clicks[s] = filseq


#-- Splitting train set and test set

print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())
if opt.dataset == 'Xing' or opt.dataset == 'Reddit':      #Split out %20 of each user's sessions as test set 
    df = pd.read_csv(dataset, delimiter=',')
    df=df.drop(['Unnamed: 0'], axis=1)
    #print(df.head())
    test_size=0.3
    test=[]
    train=[]
    #print(df['user'].max())
    for i in range(df['user'].max()):
        #if i%3000==0:
            #print(i)
        dd=df[df['user']==i]
        minimum=dd['session_id'].min()
        if minimum==0:
            minimum=minimum-1 
        maximum=dd['session_id'].max()
        lenght=maximum-minimum+1
        splitpoint=int(lenght*test_size)
        if minimum==0:
            for j in range(maximum-minimum-splitpoint):
                train.append(dd[dd['session_id']==minimum+j].values.tolist())
            for j in range(splitpoint):
                test.append(dd[dd['session_id']==maximum-j].values.tolist())
        else:
            for j in range(maximum-minimum-splitpoint+1):
                train.append(dd[dd['session_id']==minimum+j].values.tolist())
            for j in range(splitpoint):
                test.append(dd[dd['session_id']==maximum-j].values.tolist())
    tt=[item for sublist in train for item in sublist]
    tt=pd.DataFrame.from_records(tt)
    trainn=tt.rename(columns={0: "ts", 1: "item",2: "session_id", 3: "user"})
    tt=[item for sublist in test for item in sublist]
    tt=pd.DataFrame.from_records(tt)
    testt=tt.rename(columns={0: "ts", 1: "item",2: "session_id", 3: "user"})
    #print(' train:', trainn, '\n test: ',testt)

elif opt.dataset == 'diginetica':     #Split out test set based on dates (7 days for test)
    df = pd.read_csv(dataset, delimiter=';')
    df=df.drop(['userId'], axis=1)
    df=df.rename(columns={"sessionId": "session_id", "itemId": "item_id", "eventdate": "ts"})
    a=[]
    for i in range(len(df)):
        a.append(time.mktime(time.strptime(df['ts'][i], '%Y-%m-%d')))
    df['ts']=a
    #print(df.head())
    maxdate=max(df['ts'])
    splitdate = maxdate - 86400 * 7
    #print(splitdate)
    testt=df[df['ts']>=splitdate].reset_index()
    trainn=df[df['ts']<splitdate].reset_index()
    #testt.sort_values(by='session_id', ascending=True, ignore_index=True)
    #print(' train:', trainn, '\n test: ',testt)

trainn.to_csv(opt.dataset+'/trainn.csv', header=True, index=False)
testt.to_csv(opt.dataset+'/testt.csv', header=True, index=False)
    
#AVG session lenght
a=0
for i in df['session_id'].unique():
    a+=len(sess_clicks[str(i)])
print('AVG session lenght:',a/len(sess_clicks))  
#statistics
print('items:',len(df['item'].unique()),' , sessions:',len(df['session_id'].unique()))

#AVG session lenght per user (if there are user ids in dataset)
if opt.dataset == 'Xing' or opt.dataset == 'Reddit':
    for i in list(user_clicks):
        sorted_clicks_user = sorted(user_clicks[i], key=operator.itemgetter((1)))
        user_clicks[i] = [c[0] for c in sorted_clicks_user]
    a=0
    for i in range(len(user_clicks)):
        a+=len(set(user_clicks[str(i)]))
    print('AVG session lenght per user:',a/len(user_clicks))
    #statistics
    print('users:',len(df['user'].unique()))


## Convert test and train sessions to sequences and labels

def Seq_without_uid(data,df,train=True):
    if train:
        df=pd.DataFrame(columns=['session_id','train_seqs'])
    else:
        df=pd.DataFrame(columns=['session_id','test_seq','test_lab'])
    sid=data['session_id'].unique()
    for en,i in enumerate(sid):
        #if en%40000==0:
            #print(en)
        s=data[data['session_id']==i]
        if train:
            to_append=pd.Series([i,s['item_id'].tolist()]).tolist()
        else:
            to_append=pd.Series([i,s['item_id'][0:-1].tolist(),s['item_id'].tolist()[-1]]).tolist()
        a_series = pd.Series(to_append, index = df.columns)
        df = df.append(a_series, ignore_index=True)
    return df


def Seq_with_uid(data,df,train=True):
    if train:
        df=pd.DataFrame(columns=['user_id','session_id','train_seqs'])
    else:
        df=pd.DataFrame(columns=['user_id','session_id','test_seq','test_lab'])
    uid=data['user'].unique()
    for en,i in enumerate(uid):
        #if en%3000==0:
            #print(en)
        u=data[data['user']==i]
        s=u['session_id'].unique()
        for j in s:
            u_s=u[u['session_id']==j]
            if train:
                to_append=pd.Series([i,j,u_s['item'].tolist()]).tolist()
            else:
                to_append=pd.Series([i,j,u_s['item'][0:-1].tolist(),u_s['item'].tolist()[-1]]).tolist()
            a_series = pd.Series(to_append, index = df.columns)
            df = df.append(a_series, ignore_index=True)
    return df


if opt.dataset == 'Xing' or opt.dataset == 'Reddit': 
    df_t=Seq_with_uid(testt,df,train=False)
    df_tr=Seq_with_uid(trainn,df,train=True)
elif opt.dataset == 'diginetica':
    df_t=Seq_without_uid(testt,df,train=False)
    df_tr=Seq_without_uid(trainn,df,train=True)

df_t.to_csv(opt.dataset+'/df_t.csv', header=True, index=False)
df_tr.to_csv(opt.dataset+'/df_tr.csv', header=True, index=False)

te_ids=df_t['session_id']
te_seqs=df_t['test_seq']
te_labs=df_t['test_lab']
#print(te_ids[1], te_seqs[1],te_labs[1])


def process_seqs(iseqs):
    out_seqs = []    
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            ids += [id]
    return out_seqs, labs, ids

def process_seqs_tes(iseqs,ilabs):
    out_seqs = [] 
    labs = []
    ids = []
    for id, seq in zip(range(len(iseqs)), iseqs):
        out_seqs += [seq]
        ids += [id]
    for id, lab in zip(range(len(ilabs)), ilabs):
        labs += [lab]
    return out_seqs, labs, ids

te_seqs, te_labs, te_ids = process_seqs_tes(df_t['test_seq'],df_t['test_lab'])  
tr_seqs, tr_labs, tr_ids = process_seqs(df_tr['train_seqs'])

tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
#print('tr_seqs Len:', len(tr_seqs))
#print('tes_seqs Len:',len(te_seqs))
print('tr_seqs[:6]:', tr_seqs[:6], '\ntr_labs[:6]:', tr_labs[:6])
print('te_seqs[:6]:', te_seqs[:6], '\nte_labs[:6]:', te_labs[:6])


if opt.dataset == 'Xing':
    if not os.path.exists('Xing'):
        os.makedirs('Xing')
    pickle.dump(tra, open('Xing/train.txt', 'wb'))
    pickle.dump(tes, open('Xing/test.txt', 'wb'))
elif opt.dataset == 'Reddit':
    if not os.path.exists('Reddit'):
        os.makedirs('Reddit')
    pickle.dump(tra, open('Reddit/train.txt', 'wb'))
    pickle.dump(tes, open('Reddit/test.txt', 'wb'))
elif opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
print('Done.')

# item_clicks
print('finding source and targets of clicks for deepwalk: it may take long time')
item_clicks=pd.DataFrame(columns=['source','target'])
for i in df['session_id'].unique():
    lenght=len(df[df['session_id']==i])
    for j in range(lenght-1):
        item_clicks=item_clicks.append({'source': df[df['session_id']==i].reset_index(drop=True)['item'][j],
                                        'target': df[df['session_id']==i].reset_index(drop=True)['item'][j+1]}, ignore_index=True)
item_clicks.drop_duplicates(keep="first",inplace=True)
item_clicks=item_clicks.reset_index(drop=True)
item_clicks.to_csv(opt.dataset+'/item_clicks.csv', header=True, index=False)
end = time.time()
print("Run time: %f s" % (end - start))
