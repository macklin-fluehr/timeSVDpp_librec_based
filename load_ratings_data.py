__author__ = 'trimi'

import numpy as np
import random
from random import randint

class LoadRatingsData:

    def read_training_data(self, path):
         with open(path, 'rb') as f:
            matrix = []
            userItems = {}
            itemUsers = {}
            max_item = []
            timestamps = []
            b = 0

            for line in f:

                row = []
                if b % 10000 == 0:
                    print 'b  = ', b
                print line
                r = line.split()
                print 'line = ', r

                user = int(r[0])
                print user
                item = int(r[1])
                print item
                rating = float(r[2])
                print rating 
                time_ = int(r[3]) * 1000
                print time_

                row.append(user)
                row.append(item)
                row.append(rating)
                row.append(time_)

                matrix.append(row)
                max_item.append(item)
                timestamps.append(time_)

                #pos events per user
                if user not in userItems:
                    userItems[user] = [(item, rating, time_)]
                else:
                    if item not in userItems[user]:
                        userItems[user].append((item, rating, time_))

                # items rated by users
                if item not in itemUsers:
                    itemUsers[item] = [(user, rating, time_)]
                else:
                    if user not in itemUsers[item]:
                        itemUsers[item].append((user, rating, time_))

                b += 1
            print '#pos_events = ', b
            min_timestamp = min(timestamps)
            max_timestamp = max(timestamps)

            print 'max item id = ', max(max_item)
            return matrix, userItems, itemUsers, min_timestamp, max_timestamp

    def create_training_testing_set(self, matrix):

        testing_set = []
        training_set = []
        user_days = {}
        random_values = random.sample(range(1,len(matrix)), 7086)

        # testing set
        for v in range(len(random_values)):
            random_v = random_values[v]
            random_row = matrix[random_v]
            testing_set.append(random_row)

        count_samples = 0
        # training set
        for i in range(len(matrix)):
            #
            if count_samples == 28406:
                return training_set, testing_set, user_days

            row = matrix[i]
            if row in testing_set:
                continue
            else:
                user_ = row[0]
                day_ = row[3]
                if user_ not in user_days:
                    user_days[user_] = [day_]
                else:
                    if day_ not in user_days[user_]:
                        user_days[user_].append(day_)

                training_set.append(row)
                count_samples += 1

        return training_set, testing_set, user_days

    def getUserItems(self, matrix):
        userItems = {}
        itemUsers = {}
        days = []

        for i in range(len(matrix)):
            r = matrix[i]
            user = r[0]
            item = r[1]
            rating = r[2]
            time_ = r[3]

            days.append(time_)

            #pos events per user
            if user not in userItems:
                userItems[user] = [(item, rating, time_)]
            else:
                if item not in userItems[user]:
                    userItems[user].append((item, rating, time_))

            # items rated by users
            if item not in itemUsers:
                itemUsers[item] = [(user, rating, time_)]
            else:
                if user not in itemUsers[item]:
                    itemUsers[item].append((user, rating, time_))

        return userItems, itemUsers, min(days), max(days)

    def main(self):
        mat, userItems, itemUsers, min_t, max_t = self.read_training_data("...\\ratings-date.txt")

        training_mat, testing_mat, userDays = self.create_training_testing_set(mat)

        training_userItems, training_itemUsers, training_min_t, training_max_t = self.getUserItems(training_mat)

        testing_userItems, testing_itemUsers, testing_min_t, testing_max_t = self.getUserItems(testing_mat)

        return training_userItems, training_itemUsers, training_min_t, training_max_t,testing_userItems, testing_itemUsers, testing_min_t, testing_max_t, min_t, max_t, testing_mat, userItems, itemUsers, training_mat, userDays
