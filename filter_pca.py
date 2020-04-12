#!/usr/bin/env python
import numpy as np
import argparse
import pandas as pd
import sys
# import tensorflow as tf
import random
from sklearn.base import BaseEstimator, RegressorMixin

class FilterMatrix():
    def __init__(self):
        self.matrix = {}
    def lookup(self,c1,c2):
        return self.matrix.get(c1,{}).get(c2)
    def set(self,c1,c2,v):
        self.matrix.setdefault(c1,{})[c2] = v


"""
objective function = sum(y_ - (mu + r_i + c_j + sum(r_ik*c_ik)))**2
err = y_ - (mu + r_i + c_j + sum(r_ik*c_ik))
grad = -1 * err
"""

class FilterPca(BaseEstimator, RegressorMixin):
    def __init__(self, feature_cnt, N1, N2, minval = None, maxval = None):
        self.feature_cnt = feature_cnt
        self.N1 = N1
        self.N2 = N2
        self.is_setup = False
        self.row_features = None
        self.col_features = None
        self.row_offsets = None
        self.col_offsets = None
        self.minval = minval
        self.maxval = maxval
    def fit(self, X, y, iter_cnt):
        """
        args:
            X = [(x,y)]
            X is a list of (row,col) indices. The row, col numbers must be >= 0.

            y = [val]
        """
        self.total_mean = mean(y)
        y = [(v - self.total_mean) for v in y]

        #column means
        col_cnts = [0] * self.N2
        col_sums = [0] * self.N2
        for (_,q),v in zip(X,y):
            col_cnts[q] += 1
            col_sums[q] += v
        self.col_means = [0] * self.N2
        for j in range(self.N2):
            if col_cnts[j] != 0:
                self.col_means[j] = col_sums[j] / float(col_cnts[j])


        #row means
        row_cnts = [0] * self.N1
        row_sums = [0] * self.N1
        for (q,_),v in zip(X,y):
            row_cnts[q] += 1
            row_sums[q] += v
        self.row_means = [0] * self.N1
        for i in range(self.N1):
            if row_cnts[i] != 0:
                self.row_means[i] = row_sums[i] / float(row_cnts[i])


        #start all col_offsets and row_offsets at half of their respective means
        if self.row_offsets is None:
            self.row_offsets = [x/2.0 for x in self.row_means]
        if self.col_offsets is None:
            self.col_offsets = [x/2.0 for x in self.col_means]

        # print(self.total_mean);
        # print(self.col_means);
        if self.row_features is None:
            self.row_features = [[np.random.normal(scale=0.01, size=1)[0] for i in range(self.N1)] for k in range(self.feature_cnt)]
        if self.col_features is None:
            self.col_features = [[np.random.normal(scale=0.01, size=1)[0] for i in range(self.N2)] for k in range(self.feature_cnt)]
        self.lrate = 0.001
        self.lam = 15
        print("starting loop")
        for it in range(iter_cnt):
            print(f"Iteration {it}")
            if it%1 == 0: #MUST: lower frequency, this will slow down a serious run
                sum_sq_err = 0
                for (i,j),v in zip(X,y):
                    sum_sq_err += (v - self.row_offsets[i] - self.col_offsets[j] - sum([self.row_features[k][i]*self.col_features[k][j] for k in range(self.feature_cnt)]))**2
                print("sum_sq_err: " + str(sum_sq_err))
                row_offset_regularization = sum(self.lam * self.row_offsets[i] ** 2 for i in range(self.N1))
                row_feature_regularization = sum(self.lam * (self.row_features[f][i] ** 2) for i in range(self.N1) for f in range(self.feature_cnt))
                col_offset_regularization = sum(self.lam * self.col_offsets[j] ** 2 for j in range(self.N2))
                col_feature_regularization = sum(self.lam * (self.col_features[f][j] ** 2) for j in range(self.N2) for f in range(self.feature_cnt))
                objective_function = sum_sq_err + row_offset_regularization + row_feature_regularization + col_offset_regularization + col_feature_regularization

                print(f"obj_fn: {objective_function}")
            for (i,j),v in zip(X,y):
                err = v - self.row_offsets[i] - self.col_offsets[j] - sum([self.row_features[k][i]*self.col_features[k][j] for k in range(self.feature_cnt)])
                self.row_offsets[i] += self.lrate * err
                self.col_offsets[j] += self.lrate * err
                for f in range(self.feature_cnt):
                    self.row_features[f][i] += self.lrate * err * self.col_features[f][j]
                    self.col_features[f][j] += self.lrate * err * self.row_features[f][i]
            #regularization penalties
            for i in range(self.N1):
                self.row_offsets[i] -= self.lrate * self.lam * self.row_offsets[i]
                for f in range(self.feature_cnt):
                    self.row_features[f][i] -= self.lrate * self.lam * self.row_features[f][i]
            for j in range(self.N2):
                self.col_offsets[j] -= self.lrate * self.lam * self.col_offsets[j]
                for f in range(self.feature_cnt):
                    self.col_features[f][j] -= self.lrate * self.lam * self.col_features[f][j]

    def predict(self, X, use_features=True):
        if use_features:
            raw_score = [self.total_mean + self.row_offsets[i] + self.col_offsets[j] + sum([self.row_features[k][i]*self.col_features[k][j] for k in range(self.feature_cnt)]) for i,j in X]
        else:
            raw_score = [self.total_mean + self.row_offsets[i] + self.col_offsets[j] for i,j in X]
        if self.minval is not None:
            raw_score = [max(self.minval, s) for s in raw_score]
        if self.maxval is not None:
            raw_score = [min(self.maxval, s) for s in raw_score]
        return raw_score
    def score(self, X, y):
        preds = self.predict(X)
        partial_preds = self.predict(X, use_features=False)
        r2 = sum((self.total_mean - val)**2 for (i,j),val in zip(X,y))
        r2_partial_resid = sum([(p - val)**2 for p,val in zip(partial_preds, y)])
        r2_resid = sum([(p - val)**2 for p,val in zip(preds, y)])
        print(1 - (float(r2_partial_resid) / float(r2)))
        print(1 - (float(r2_resid) / float(r2)))

    def to_csv(self, row_idx2name = None, col_idx2name = None, stdout=False):
        # row_features, col_features = pred.row_features, pred.col_features #train(N1, N2, vals, feature_cnt, iter_cnt=iterations)
        if not row_idx2name:
            row_idx2name = dict((i,i) for i in range(self.N1))
        if not col_idx2name:
            col_idx2name = dict((i,i) for i in range(self.N2))
        SORT_BY = "feature_0"

        user_cols = ["user","offset"] + ["feature_"+str(k) for k,_ in enumerate(self.row_features)]
        df_users = pd.DataFrame([dict([("user",row_idx2name[i]), ("offset",self.row_offsets[i])] + [("feature_"+str(k),f) for k,f in enumerate(features)]) for i,features in enumerate(zip(*self.row_features))],columns=user_cols).sort_values(by=SORT_BY)

        item_cols = ["item","offset"] + ["feature_"+str(k) for k,_ in enumerate(self.col_features)]
        df_items = pd.DataFrame([dict([("items",col_idx2name[i]), ("offset",self.col_offsets[i])] + [("feature_"+str(k),f) for k,f in enumerate(features)]) for i,features in enumerate(zip(*self.col_features))], columns=item_cols).sort_values(by=SORT_BY)

        #replace small values with 0
        for i,c in enumerate([c for c in df_users.columns if not c == "user"]):
            try:
                df_users.loc[abs(df_users[c]) < 1e-3,c] = 0
            except Exception as e:
                raise e
        for i,c in enumerate([c for c in df_items.columns if not c == "item"]):
            df_items.loc[abs(df_items[c]) < 1e-3,c] = 0

        # df_users.to_csv(sys.stdout, index=False)
        # print "---------------------\n\n\n"
        # df_items.to_csv(sys.stdout, index=False)
        if stdout:
            df_users.to_csv(sys.stdout, index=False)
            df_items.to_csv(sys.stdout, index=False)
        else:
            df_users.to_csv("/tmp/pca_users.csv", index=False)
            df_items.to_csv("/tmp/pca_items.csv", index=False)



def train(N1, N2, vals, feature_cnt, iter_cnt=10000):
    row_features = [[np.random.normal(scale=0.1, size=1)[0] for i in range(N1)] for k in range(feature_cnt)]
    col_features = [[np.random.normal(scale=0.1, size=1)[0] for i in range(N2)] for k in range(feature_cnt)]
    lrate = 0.001
    for it in range(iter_cnt):
        print(f"Iteration {it}")
        if it%100 == 0:
            total_err = 0
            for i,j,v in vals:
                total_err += abs(v - sum([row_features[k][i]*col_features[k][j] for k in range(feature_cnt)]))
            print("total_err: " + str(total_err))
        for f in range(feature_cnt):
            for i,j,v in vals:
                err = v - sum([row_features[k][i]*col_features[k][j] for k in range(feature_cnt)])
                row_val = row_features[f][i]
                col_val = col_features[f][j]
                row_features[f][i] += lrate * err * col_val
                col_features[f][j] += lrate * err * row_val
    return row_features, col_features


def test(i,j,v):
    N1 = 3
    N2 = 3
    mean = 0
    row_means = [0] * N1
    col_means = [0] * N2
    feature_cnt = 1
    err = (v - (mean + row_means[i] + col_means[j] + sum([k1*k2 for k1,k2 in zip(row_factors[i],col_factors[j])])))
    #scipy.optimize.minimize(fn1, data1, jac=grad_fn1, args=tuple(args_dict.values()), method=method, bounds=bounds1, options=options1)

def readCL():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--infile")
    parser.add_argument("-i","--iterations",type=int, default=5000)
    args = parser.parse_args()
    return args.infile, args.iterations

def test_filter_pca():
    N1 = 5
    N2 = 5
    feature_cnt = 2
    vals = [(i,j,i*j + np.random.normal(scale=0.1, size=1)[0]) for i in range(N1) for j in range(N2)]
    X = [v[:2] for v in vals]
    y = [v[2] for v in vals]
    pred = FilterPca(feature_cnt=feature_cnt, N1=N1, N2=N2)
    pred.fit(X,y,1000)
    pred.to_csv(stdout=True)
    # row_features, col_features = tf_train(N1, N2, vals, feature_cnt, 10000)
    # print row_features, col_features



def tf_train(N1, N2, vals, feature_cnt, iter_cnt):
    #some pieces from here:
    #http://katbailey.github.io/post/matrix-factorization-with-tensorflow/

    item_row = tf.placeholder(tf.int32, [None])
    item_col = tf.placeholder(tf.int32, [None])
    y_ = tf.placeholder(tf.float32, [None])

    total_mean = tf.Variable(tf.random_normal([],stddev=0.1))
    row_offsets = tf.Variable(tf.random_normal([N1], stddev=0.1))
    col_offsets = tf.Variable(tf.random_normal([N2], stddev=0.1))
    row_features = tf.Variable(tf.random_normal([N1, feature_cnt], stddev=0.1))
    col_features = tf.Variable(tf.random_normal([N2, feature_cnt], stddev=0.1))
    regularization_penalty = tf.constant(0e-4)

    item_row_offset = tf.gather(row_offsets, item_row)
    item_col_offset = tf.gather(col_offsets, item_col)

    item_row_features = tf.gather(row_features, item_row)
    item_col_features = tf.gather(col_features, item_col)
    # item_row_features = tf.slice(row_features, [item_row,0], [1, -1])
    # item_col_features = tf.slice(col_features, [item_col,0], [1, -1])
    # item_row_features = tf.gather(tf.reshape(result, [-1]), user_indices * tf.shape(result)[1] + item_indices, name="extract_training_ratings")

    # row_features = tf.Print(row_features,[row_features], "row_features: ", summarize=1000)
    # item_row_features = tf.Print(item_row_features,[item_row_features], "item_row_features: ")

    feature_prod = tf.diag_part(tf.matmul(item_row_features, item_col_features, transpose_b = True))
    pred = tf.add_n([tf.fill(tf.shape(item_row), total_mean), item_row_offset, item_col_offset, feature_prod])
    error = tf.reduce_mean(tf.square(tf.sub(pred, y_)))
    # reg_penalty = tf.mul(regularization_penalty,
    #                          tf.nn.l2_loss(tf.concat(0,[tf.expand_dims(total_mean, 0),
    #                                                     tf.reshape(row_offsets, [-1]),
    #                                                     tf.reshape(col_offsets, [-1]),
    #                                                     tf.reshape(row_features, [-1]),
    #                                                     tf.reshape(col_features, [-1])])))

    reg_penalty = tf.mul(regularization_penalty,
                             tf.reduce_sum(tf.abs((tf.concat(0,[tf.expand_dims(total_mean, 0),
                                                        tf.reshape(row_offsets, [-1]),
                                                        tf.reshape(col_offsets, [-1]),
                                                        tf.reshape(row_features, [-1]),
                                                        tf.reshape(col_features, [-1])])))))


    cost = tf.add(error, reg_penalty)
    cost_summary = tf.scalar_summary("cost", cost)

    tf.scalar_summary("reg_penalty",reg_penalty)
    tf.scalar_summary("error",error)
    tf.scalar_summary("cost",cost)

    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/recommender_logs", sess.graph)

    print(len(vals))
    for i in range(iter_cnt):
        training_set, test_set = gen_train_test(vals, 0.8)
        item_row_val, item_col_val, y_val = training_set[i % len(training_set)]
        _, cost_val, error_val = sess.run([train_step, cost, error], {item_row: [item_row_val],
                                               item_col: [item_col_val],
                                               y_: [y_val]})
        if i % 500 == 0:
            print("training cost/error: " + str(cost_val) + "/" + str(error_val))
            rows, cols, labels = zip(*test_set)
            out = sess.run([total_mean, row_offsets, col_offsets, row_features, col_features, pred, reg_penalty, cost, error],
                           {item_row: rows,
                            item_col: cols,
                            y_: labels})
            # print("info: " + str(out))
            # print("pred: " + str(out[5]))
            # print("reg: " + str(out[6]))
            # print("cost: " + str(out[7]))
            print("test error: " + str(out[8]))
            # print("info: " + str([out[0],out[5]]))
    out = sess.run([row_features, col_features, row_offsets, col_offsets])
    row_features_out, col_features_out, row_offsets_out, col_offsets_out = out
    return [row_offsets_out] + zip(*row_features_out), [col_offsets_out] + zip(*col_features_out)

def gen_train_test(l, training_prob):
    train = []
    test = []
    for i in l:
        if random.random() < training_prob:
            train.append(i)
        else:
            test.append(i)
    return train, test


def mean(l):
    return sum(l) / float(len(l))

def preprocess():
    infile, iterations = readCL()
    if not infile: raise
    print("loading...")
    df = pd.read_csv(infile)
    all_users = list(set(df.iloc[:,0].values))
    all_items = list(set(df.iloc[:,1].values))
    # items_to_keep = ["q156917", "q210", "q118236", "q42"]
    # items_to_keep = set(random.sample(all_items,1000))
    # users_to_keep = set(random.sample(all_users,1000))
    items_to_keep = set(all_items)
    users_to_keep = set(all_users)

    print("filtering")
    df = df[df.apply(lambda x: x[0] in users_to_keep and x[1] in items_to_keep, axis=1)]
    print("done filtering")
    # df.to_csv("/tmp/a.csv", index=False)
    # raise
    users = list(set(df.iloc[:,0].values))
    items = list(set(df.iloc[:,1].values))
    user2idx = dict((a,i) for i,a in enumerate(users))
    item2idx = dict((q,i) for i,q in enumerate(items))
    idx2user = dict((i,a) for i,a in enumerate(users))
    idx2item = dict((i,q) for i,q in enumerate(items))
    N1 = len(users)
    N2 = len(items)
    vals = [[user2idx[a], item2idx[q], v] for a,q,v in df.values]
    vals_train, vals_test = gen_train_test(vals, training_prob = 0.8)
    X_train = [v[:2] for v in vals_train]
    y_train = [v[2] for v in vals_train]
    X_test = [v[:2] for v in vals_test]
    y_test = [v[2] for v in vals_test]
    return idx2user, idx2item, N1, N2, X_train, y_train, X_test, y_test

def main():
    # total_mean = mean([v for _,_,v in vals])
    # vals = [(i,j,v-total_mean) for i,j,v in vals]
    # item_means = dict((j,mean([v for _,q,v in vals if q == j])) for j in range(N2))
    # vals = [(i,j,v-item_means[j]) for i,j,v in vals]
    idx2user, idx2item, N1, N2, X_train, y_train, X_test, y_test = preprocess()
    feature_cnt = 2
    MIN_VAL = -1
    MAX_VAL = 1
    pred = FilterPca(feature_cnt=feature_cnt, N1=N1, N2=N2, minval=MIN_VAL, maxval=MAX_VAL)
    print("prefit")
    for i in range(10):
        pred.fit(X_train, y_train, 100)
        pred.score(X_train, y_train)
        pred.score(X_test, y_test)
        print(sum(abs(x) for x in pred.col_features[0]))
        print(sum(abs(x) for x in pred.row_features[0]))
        pred.to_csv(idx2user, idx2item)

if __name__ == "__main__":
    main()
    # test_filter_pca()
