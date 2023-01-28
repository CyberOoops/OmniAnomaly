import ast
import csv
import os
import sys
from pickle import dump

import numpy as np
from tfsnippet.utils import makedirs

output_folder = 'processed'
makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder = 'ServerMachineDataset'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)
                
    elif dataset == "SWAT":
        train_path = "../datasets/SWAT/SWaT_Dataset_Normal_v1.csv"
        test_path = "../datasets/SWAT/SWaT_Dataset_Attack_v0.csv"
        
        with open(train_path, 'r')as file:
            csv_reader = csv.reader(file, delimiter=',')
            res_train = [row[1:-1] for row in csv_reader][2:]
            row_train = len(res_train)
            traindata = np.array(res_train, dtype=np.float32)[21600:]
            print(traindata.shape)
            
        # traindata = np.delete(traindata, [5,10], axis=1)
        # data_all = np.concatenate((traindata, testdata), axis=0)
        epsilo = 0.001
        data_min = np.min(traindata, axis=0)
        data_max = np.max(traindata, axis=0)+epsilo
        for i in range(len(data_max)):
            if data_max[i] - data_min[i] < 10 * epsilo:
                data_min[i] = data_max[i]
                data_max[i] = 1 + data_max[i]
                
        mu = np.mean(traindata, axis=0)
        sigma = np.std(traindata, axis=0)
        epsilo = 0.01
        for i in range(len(sigma)):
            if sigma[i] < epsilo:
                sigma[i] = 1
        
        train_ = (traindata - data_min)/(data_max - data_min)
            # rawdata = (traindata - mu) / sigma
        print("train shape ", train_.shape)
                
        with open(test_path, 'r')as file:
            csv_reader = csv.reader(file, delimiter=',')
            res_test = [row[1:-1] for row in csv_reader][1:]
            row_test = len(res_test)
            testdata = np.array(res_test, dtype=np.float32)
            # testdata = np.delete(testdata, [5,10], axis=1)
            print(testdata.shape)
            
        test_ = (testdata - data_min)/(data_max - data_min)
        # rawdata = (testdata - mu) / sigma
        print("test shape ", test_.shape)
            
        test_ = np.clip(test_, a_min=-1.0, a_max=3.0)
        
        label_path = "../datasets/SWAT/SWaT_Dataset_Attack_v0.csv"
        with open(label_path, 'r')as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row[-1]for row in csv_reader][1:]
            label_ = [0 if i == "Normal" else 1 for i in res]
            label_ = np.array(label_)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(label_, file)
        with open(os.path.join(output_folder, dataset + "_" + 'test' + ".pkl"), "wb") as file:
            dump(test_, file)
        with open(os.path.join(output_folder, dataset + "_" + 'train' + ".pkl"), "wb") as file:
            dump(train_, file)
        
        
        
    elif dataset == "WADI":
        nan_cols = []
        train_path = "../datasets/WADIA2/WADI_14days_new.csv"
        
        
        with open(train_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')
            res_train = [row[3:] for row in csv_reader][1:]
            res_train = np.array(res_train)[21600:]
            row_train, col_train = len(res_train), len(res_train[0])
            for j in range(res_train.shape[1]):
                for i in range(res_train.shape[0]):
                    if res_train[i][j] == "1.#QNAN" or res_train[i][j] == '':
                        nan_cols.append(j)
                        break
            # len(nan_cols) == 9
            res_train = np.delete(res_train, nan_cols, axis=1)
            res_train = res_train.astype(np.float32)
                        
        traindata = res_train
                        
        test_path = "../datasets/WADIA2/WADI_attackdataLABLE.csv"
        
        epsilo = 0.001
        data_min = np.min(traindata, axis=0)
        data_max = np.max(traindata, axis=0)+epsilo
        for i in range(len(data_max)):
            if data_max[i] - data_min[i] < 10 * epsilo:
                data_min[i] = data_max[i]
                data_max[i] = 1 + data_max[i]
        traindata = (traindata - data_min)/(data_max - data_min)
                        
        train_ = traindata
        print("train shape ", train_.shape)
                
        with open(test_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res_test = [row[3:-1] for row in csv_reader][2:]
            res_test = np.array(res_test)
            row_test, col_test = len(res_test), len(res_test[0])
            for i in range(row_test):
                for j in range(col_test):
                    if res_test[i][j] == '':
                        res_test[i][j] = 0
            res_test = np.delete(res_test, nan_cols, axis=1)
            res_test = res_test.astype(np.float32)            
        test_ = (res_test - data_min)/(data_max - data_min)
        print("test shape ", test_.shape)
            
        test_ = np.clip(test_, a_min=-1.0, a_max=2.0)
        
        label_path = "../datasets/WADIA2/WADI_attackdataLABLE.csv"
        with open(label_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row[-1] for row in csv_reader][2:]
            label_ = np.array(res, dtype=np.float32)
            for i in range(len(label_)):
                if label_[i] <= 0:
                    label_[i] = 1
                else:
                    label_[i] = 0
        
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(label_, file)
        with open(os.path.join(output_folder, dataset + "_" + 'test' + ".pkl"), "wb") as file:
            dump(test_, file)
        with open(os.path.join(output_folder, dataset + "_" + 'train' + ".pkl"), "wb") as file:
            dump(train_, file)
            
    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = 'data'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ['train', 'test']:
            concatenate_and_save(c)


if __name__ == '__main__':
    datasets = ['SMD', 'WADI', 'SWAT']
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """)
