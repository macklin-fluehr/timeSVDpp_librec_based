__author__ = 'trimi'

import numpy as np
import math
import csv


from load_ratings_data import LoadRatingsData
class timeSVDpp:

    def __init__(self, iter, nFactors, nUsers, nItems, userItems, nBins, nDays, testing_mat, training_userItems, training_mat, userDays): #  min_time_in_seconds

    # def __init__(self, iter, nFactors, nUsers, nItems, userItems, nBins, nDays, min_time_in_seconds, timestamps): #  min_time_in_seconds
        self.gamma_1 = 0.005
        self.gamma_2 = 0.007
        self.gamma_3 = 0.001
        self.g_alpha = 0.00001
        self.tau_6 = 0.005
        self.tau_7 = 0.015
        self.tau_8 = 0.015
        self.l_alpha = 0.0004

        # self.timestamps = timestamps
        self.max_time = nDays
        # self.min_time = 1896
        self.min_time = 0
        # self.min_time_in_seconds = min_time_in_seconds
        self.min_time_in_seconds = 1000
        # self.userDays = userDays

        self.iterations = iter
        self.userItems = userItems
        self.testing_mat = testing_mat
        self.training_userItems = training_userItems
        self.training_mat = training_mat

        self.factors = nFactors + 1
        self.nUsers = nUsers + 1
        self.nItems = nItems + 1
        self.nBins = nBins
        #self.nDays = 214
        # for #users = 30, 1963 days
        self.nDays = nDays + 1

        #initialization
        print('initialization started...')
        b_u, b_i, u_f, i_f, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t, cu, cu_t = self.init(self.nUsers, self.nItems, self.factors, self.nBins)

        self.bu = b_u
        self.bi = b_i
        self.bi_bin = bi_bin
        self.alpha_u = alpha_u
        self.bu_t = bu_t

        self.cu = cu
        self.cu_t = cu_t

        self.userFactors = u_f
        self.itemFactors = i_f
        self.y_j = y_j
        self.sumMW = sumMW

        self.alpha_u_k = alpha_u_k
        self.userFactors_t = userFactors_t
        print 'initialization finished...'

        self.average = self.avg()
        print 'avg = ', self.average

        print 'training started...'
        self.train(self.iterations)
        print 'training finished...'

        print 'evaluation started...'
        rmse = self.RMSE_librec()
        print 'RMSE = ', rmse
        print 'evaluation finished'
        # test

    def init(self, nUsers, nItems, nFactors, nBins):
        #biases
        bu = np.random.uniform(0, 1, nUsers + 1)
        bi = np.random.uniform(0, 1, nItems + 1) # dtype = 'float64'

        # bi_bin = np.random.random((nItems + 1, nBins))
        bi_bin = []
        for b in range(nItems + 1):
            bii = np.random.uniform(0, 1, nBins)
            bi_bin.append(bii)

        alpha_u = np.random.uniform(0, 1, nUsers + 1) # nUsers + 1, dtype = 'float64'

        bu_t = np.zeros((nUsers + 1, self.nDays), dtype = 'float64')
        """for us in range(1, len(bu_t)):
            if self.userDays[us] > 0:
                for g in range(len(self.userDays)):
                    day_ = self.userDays[us][g]
                    bu_t[us][day_] = np.random.uniform(0, 1)"""

        cu = np.random.uniform(0, 1, nUsers + 1)
        cu_t = []
        for b in range(nItems + 1):
            cuu = np.random.uniform(0, 1, self.nDays)
            cu_t.append(cuu)

        #factors
        # userFactors = np.random.random((nUsers + 1, nFactors))
        userFactors = []
        for b in range(nUsers + 1):
            bii = np.random.uniform(0, 1, nFactors)
            userFactors.append(bii)

        # itemFactors = np.random.random((nItems + 1, nFactors))
        itemFactors = []
        for b in range(nItems + 1):
            bii = np.random.uniform(0, 1, nFactors)
            itemFactors.append(bii)

        # y_j = np.random.random((nItems + 1, nFactors), np.float64)
        y_j = []
        for b in range(nItems + 1):
            bii = np.random.uniform(0, 1, nFactors)
            y_j.append(bii)

        # sumMW = np.random.random((nUsers + 1, nFactors))
        sumMW = []
        for b in range(nUsers + 1):
            bii = np.random.random(nFactors)
            sumMW.append(bii)

        #time-based parameters
        # alpha_u_k = np.random.random((nUsers + 1, nFactors))
        alpha_u_k = []
        for b in range(nUsers + 1):
            bii = np.random.uniform(0, 1, nFactors)
            alpha_u_k.append(bii)

        userFactors_t = np.zeros((nUsers + 1, nFactors, self.nDays))
        """for userr in range(1, len(userFactors_t)):
            if self.userDays[userr] > 0:
                for da in range(len(self.userDays)):
                    day_ = self.userDays[userr][da]
                    for fct in range(nFactors):
                        userFactors_t[userr][fct][day_] = np.random.uniform(0, 1)"""

        return bu, bi, userFactors, itemFactors, y_j, sumMW, bi_bin, alpha_u, bu_t, alpha_u_k, userFactors_t, cu, cu_t

    def train(self, iter):
        for i in range(iter):
            loss = 0
            print ('-------------------', i + 1,' ----------------')
            self.oneIteration()
            rmse = self.RMSE_librec()
            print 'iteration: ', i + 1, ', RMSE = ', rmse

    def oneIteration(self):
        loss = 0
        for userId, v in self.training_userItems.items():
            # print 'userID = ', userId
            if userId % 500 == 0:
                print 'users trained: ', userId
            tmpSum = np.zeros(self.factors, dtype='float')

            # if userId not in self.training_userItems:
             #   continue
            sz = len(self.training_userItems[userId])

            if sz > 0:

                for it in range(len(self.training_userItems[userId])):
                    itemid = self.training_userItems[userId][it][0]
                    rating = self.training_userItems[userId][it][1]
                    timestamp_ = self.training_userItems[userId][it][2]

                    # li = [userId, itemid, rating, timestamp_]
                    # if li not in self.training_mat:
                    #     continue

                    sqrtNum = 1/(math.sqrt(sz))

                    day_ind = int((timestamp_ - self.min_time_in_seconds)/86400000)
                    cu_ = self.cu[userId]
                    cut_ = self.cu_t[userId][day_ind]
                    bi_ = self.bi[itemid]
                    bit_ =  self.bi_bin[itemid][self.calBin(day_ind)]
                    bu_ = self.bu[userId]
                    but_ = self.bu_t[userId][day_ind]
                    au = self.alpha_u[userId]
                    dev_ = self.dev(userId, day_ind)


                    sy = []
                    for a in range(self.factors):
                        res = 0
                        for it in range(sz):
                            item_id_ = self.training_userItems[userId][it][0]
                            res += self.y_j[item_id_][a]
                        res = sqrtNum * res
                        sy.append(res)


                    pred = self.average + (bi_ + bit_) * (cu_ + cut_)
                    pred += bu_ + au * dev_ + but_

                    for f in range(self.factors):
                        qik = self.itemFactors[itemid][f]
                        sy_ = sy[f]
                        pred += sy_ * qik

                    for f in range(self.factors):
                        qik = self.itemFactors[itemid][f]
                        puk = self.userFactors[userId][f] + self.alpha_u_k[userId][f] * self.dev(userId, day_ind) + self.userFactors_t[userId][f][day_ind]
                        pred += puk * qik

                    prediction = self.prediction(userId, itemid, day_ind)
                    error = pred - rating
                    # loss += error * error
                   #  print 'prediction = ', prediction
                    # print 'error = ', error
                    # user bias
                    sgd = error +  0.01 * bu_
                    self.bu[userId] += -0.01 * sgd
                    # loss += 0.01 * bu_ * bu_

                    # item bias
                    sgd = error * (cu_ + cut_) + 0.01 * bi_
                    self.bi[itemid] += -0.01 * sgd
                    # loss += 0.01 * bi_ * bi_

                    # item bias bi, bin(t)
                    sgd = error * (cu_ + cut_) + 0.01 * bit_
                    self.bi_bin[itemid][self.calBin(day_ind)] += -0.01 * sgd
                    # loss += 0.01 * bit_ + bit_
                    # cu
                    sgd = error * (bi_ + bit_) + 0.01 * cu_
                    self.cu[userId] += -0.01 * sgd

                    #cu_t
                    sgd = error * (bi_ + bit_) + 0.01 * cut_
                    self.cu_t[userId][day_ind] += -0.01 * sgd

                    # bu_t
                    sgd = error + 0.01 * but_
                    delta = but_ - 0.01 * sgd
                    self.bu_t[userId][day_ind] = delta
                    # loss += 0.01 * but_ + but_

                    # au
                    sgd = error * dev_ + 0.01 * au
                    self.alpha_u[userId] += -0.01 * sgd
                    # loss += 0.01 * au + au

                    # print bi_, bit_, bu_, but_, au, dev_

                    # updating factors
                    for k in range(self.factors):
                        u_f = self.userFactors[userId][k]
                        i_f = self.itemFactors[itemid][k]
                        u_f_t = self.userFactors_t[userId][k][day_ind]
                        auk = self.alpha_u_k[userId][k]



                        pukt = u_f + auk * dev_ + u_f_t

                        # print u_f, auk, dev_, u_f_t
                        # print pukt

                        sum_yk = 0
                        for j in range(sz):
                            pos_item = self.training_userItems[userId][j][0]
                            sum_yk += self.y_j[pos_item][k]

                        # print u_f, i_f, u_f_t, auk, sum_yk

                        sgd = error * (pukt + sqrtNum * sum_yk) + 0.01 * i_f
                        self.itemFactors[itemid][k] += -0.01 * sgd
                       #  loss += 0.01 * i_f * i_f

                        #update user factor
                        sgd = error * i_f + 0.01 * u_f
                        self.userFactors[userId][k] += -0.01 * sgd
                        # loss += 0.01 * u_f * u_f

                        # auk
                        sgd  = error * i_f * dev_ + 0.01 * auk
                        self.alpha_u_k[userId][k] += -0.01 * sgd
                        # loss += 0.01 * auk * auk

                        # uf_t
                        sgd = error * i_f + 0.01 * u_f_t
                        delta = u_f_t - 0.01 * sgd
                        self.userFactors_t[userId][k][day_ind] = delta
                        # loss += 0.01 * u_f_t * u_f_t

                        for j in range(sz):
                            itID = self.training_userItems[userId][j][0]
                            yjk_ = self.y_j[itID][k]
                            sgd = error * sqrtNum * i_f + 0.01 * yjk_
                            self.y_j[itID][k] += -0.01 * sgd
                            # loss += 0.01 * yjk_ * yjk_
        # loss *= 0.5
        # print 'loss = ', loss

    #overall rating avarage
    def avg(self):
        s = 0
        count = 0

        for i, v in self.training_userItems.items():
            #if i not in self.training_userItems:
             #   continue
            sz = len(self.training_userItems[i])
            for j in range(sz):
                rating_ = self.training_userItems[i][j][1]
                s += rating_
                count += 1
        avg = s/count

        return avg

    #find the index of the bin for the given timestamp
    def calBin(self, day_of_rating):
        interval = (self.max_time - 0.0) / self.nBins
        bin_ind = min(self.nBins - 1, int((day_of_rating - self.min_time)/interval))

        return bin_ind

    #deviation of user u at given t
    def dev(self, userID, t):
        deviation = np.sign(t - self.meanTime(userID)) * pow(abs(t - self.meanTime(userID)), 1)

        return deviation

    #mean rating time for given user
    def meanTime(self, userID):
        s = 0
        count = 0
        sz = len(self.training_userItems[userID])
        if sz > 0:
            list_of_days = []
            for i in range(sz):
                timestamp__ = self.training_userItems[userID][i][2]
                d_ind = int((timestamp__ - self.min_time_in_seconds)/86400000)
                # if d_ind not in list_of_days:
                    # list_of_days.append(d_ind)
                s += d_ind
                count += 1

            return s/count
        else:
            summ = 0
            l_of_days = []
            cc = 0
            for i in range(len(self.timestamps)):
                dind = int((self.timestamps[i] - self.min_time_in_seconds)/86400000)
                if dind not in l_of_days:
                    l_of_days.append(dind)
                    summ += dind
                    cc += 1
            globalMean = summ/cc

            return globalMean

    #prediction method
    def prediction(self, user_id, item_id, day_ind):

        if user_id in self.training_userItems:
            sz = len(self.training_userItems[user_id])
            sqrtNum = 1/(math.sqrt(sz))
        else:
            sz = len(self.training_userItems[user_id])
            sqrtNum = 1/(math.sqrt(sz))
            print 'user not trained...'

        # global mean
        prediction = self.average

        # item bias
        prediction += (self.bi[item_id] + self.bi_bin[item_id][self.calBin(day_ind)]) * (self.cu[user_id] + self.cu_t[user_id][day_ind])

        # user bias
        prediction += self.bu[user_id] + self.alpha_u[user_id] * self.dev(user_id, day_ind) + self.bu_t[user_id][day_ind]


        # sum of the features from the Ru set
        sy = []
        for a in range(self.factors):
            res = 0
            for it in range(sz):
                item_id_ = self.training_userItems[user_id][it][0]
                res += self.y_j[item_id_][a]
            res = sqrtNum * res
            sy.append(res)

        # dot product between the features of the item and the summed features from the Ru set
        for f in range(self.factors):
            qik = self.itemFactors[item_id][f]
            sy_ = sy[f]
            prediction += sy_ * qik

        # dot product between user features and item features
        for f in range(self.factors):
            qik = self.itemFactors[item_id][f]
            pukt = self.userFactors[user_id][f] + self.alpha_u_k[user_id][f] * self.dev(user_id, day_ind) + self.userFactors_t[user_id][f][day_ind]
            prediction += pukt * qik


        return prediction


    def RMSE_librec(self):

        mean_squared_error = 0
        c = 0

        for i in range(len(self.testing_mat)):
            row = self.testing_mat[i]
            userid = row[0]
            itemid = row[1]
            rating = float(row[2])
            t_stamp = int(row[3])
            counting = 0
            day = int((t_stamp - self.min_time_in_seconds)/ 86400000)
            if userid not in self.training_userItems:
                counting += 1
                continue

            predict = self.prediction(userid, itemid, day)

            mean_squared_error += math.pow((rating - predict), 2)

            c += 1

        meanSuaredError = mean_squared_error/c
        meanSuaredError = math.sqrt(meanSuaredError)
        print 'counting: ', counting
        return meanSuaredError

# M A I N


# librec data
lr = LoadRatingsData()
training_userItems, training_itemUsers, training_min_t, training_max_t, testing_userItems, testing_itemUsers, testing_min_t, testing_max_t, min_t, max_t, testing_mat, total_userItems, total_itemUsers, training_mat, userDays = lr.main()

nFactors = 10
nBins = 6

nUsers = len(total_userItems)
nItems = len(total_itemUsers)

ccc = 0
for l in range(len(training_mat)):
    a = training_mat[l]
    if a == []:
        ccc += 1
print 'ccc = ', ccc

userItems = training_userItems
nDays = max_t

print len(training_userItems), len(training_itemUsers)

timesvd_pp = timeSVDpp(100, nFactors, nUsers, nItems, total_userItems, nBins, 1, testing_mat, training_userItems, training_mat, userDays)

#10, bins, 1 iteration: RMSE = 1.02648993887
#10 bins,  20 iterations: RMSE = 0.94744760782
# 10 bins, 30 iterations: RMSE = 0.94331981365

# 30 bins, 20 iterations: 0.94672648969

# iteration:  20 , RMSE =  0.743524108821