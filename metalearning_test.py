import copy
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from sklearn.model_selection import train_test_split
import math
import random

ratio = 0.8
feature_num = 0
class_num = 2
task_num = 4
curr_task = 0

# 测试任务的数量
num_test_task = 1

alpha = []
beta = 0.001

csv_file = ['processed_Dos_data.csv','processed_Fuzzy_data.csv','processed_gear_data.csv','processed_rpm_data.csv']

# 存储分割产生的任务集用
tasks_dataset = []

def file_exists(filename):
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)

    if filename in files_in_dir:
        return True
    else:
        return False

# 根据任务数量分割数据集
def split_tasks_datasets(task_num):
    global tasks_dataset

    for i in range(task_num):
        if file_exists('x_data'+str(i)+'.pt') and file_exists('y_data'+str(i)+'.pt'):
            pass
        else:
            category = 1
            # df_dos = pd.read_csv('processed_normal_run_data.csv')

            for attack_dataset_file in csv_file:
                df = pd.read_csv(attack_dataset_file)
                df.loc[:, ['Label']] = df.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else category)
                category += 1
                groups = df['Label'].unique()
                # 遍历唯一值
                for group in groups:
                    # 过滤出该组数据
                    df_group = df[df['Label'] == group]
                    # 获取组内总行数
                    rows = len(df_group)
                    # 计算每份的行数
                    rows_per_split = rows // task_num

                    # 分割数据集进每个任务
                    if len(tasks_dataset) == 0:
                        for i in range(task_num):
                            if i ==0:
                                tasks_dataset.append(df_group.iloc[:rows_per_split])
                            elif i == task_num-1:
                                tasks_dataset.append(df_group[rows_per_split*i:])
                            else:
                                tasks_dataset.append(df_group.iloc[rows_per_split*i:rows_per_split*(i+1)])
                    else:
                        for i in range(task_num):
                            if i == 0:
                                tasks_dataset[i] = pd.concat([tasks_dataset[i], df_group.iloc[:rows_per_split]])
                            elif i == task_num-1:
                                tasks_dataset[i] = pd.concat([tasks_dataset[i],df_group.iloc[rows_per_split*i:]])
                            else:
                                tasks_dataset[i] = pd.concat([tasks_dataset[i],df_group.iloc[rows_per_split*i:rows_per_split*(i+1)]])
            # 将分割出的每个任务数据集保存
            for i in range(task_num):
                xy = tasks_dataset[i].iloc[:,1:]
                x_data = torch.Tensor(xy.iloc[:xy.shape[0], :-1].values)
                y_data = torch.Tensor(xy.iloc[:xy.shape[0],[-1]].values.astype(float))

                torch.save(x_data,'x_data'+str(i)+'.pt')
                torch.save(y_data,'y_data'+str(i)+'.pt')

# 为meta-sgd定义sgd优化函数
def sgd_optimize(paralist, lrlist, gradlist):
    for para,lr,grad in zip(paralist,lrlist,gradlist):
        para.data -= lr * grad












# 封装每个任务的数据
class Traffic_Task_Dataset(Dataset):

    def __init__(self,no_task):
        global feature_num
        if file_exists('x_data{}.pt'.format(no_task)) and file_exists('y_data{}.pt'.format(no_task)):
            self.x_data = torch.load('x_data{}.pt'.format(no_task))
            self.y_data = torch.load('y_data{}.pt'.format(no_task))
            self.n_samples = len(self.x_data)
            feature_num = self.x_data.shape[1]
        else:
            print("Did not find the dataset files of tasks")


    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]


    def __len__(self):
        return self.n_samples


# class Traffic_Test_Dataset(Dataset):
#
#     def __init__(self):
#         df_normal = pd.read_csv('processed_normal_run_data.csv')
#         df_dos = pd.read_csv('processed_Dos_data.csv')
#         xy = pd.concat([df_normal, df_dos])
#
#         xy = xy.iloc[:, 1:]
#
#         xy.loc[:, ['Label']] = xy.loc[:, ['Label']].applymap(lambda x: 0 if x == 'R' else 1)
#
#         self.train_samples = math.floor(xy.shape[0] * 0.8)
#
#
#         self.x_data = torch.Tensor(xy.iloc[self.train_samples:,:-1].values)
#
#         self.y_data = torch.Tensor(xy.iloc[self.train_samples:,[-1]].values.astype(int))
#
#         self.n_samples = xy.shape[0] - math.floor(xy.shape[0] * 0.8)
#
#
#     def __getitem__(self, index):
#         return self.x_data[index],self.y_data[index]
#
#
#     def __len__(self):
#         return self.n_samples




class LSTM_RNN(nn.Module):
    def __init__(self):
        super(LSTM_RNN,self).__init__()
        self.lstm = nn.LSTM(feature_num,feature_num*2,num_layers=1,batch_first=True)
        self.norm = nn.BatchNorm1d(feature_num*2)
        self.linear1 = nn.Linear(feature_num*2, feature_num)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(feature_num,math.ceil(feature_num/2))
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(math.ceil(feature_num/2), 2)


    def forward(self,x):
        x = x.view(-1,1,feature_num)
        lstm_out, (h_n, h_c) = self.lstm(x)
        b,s,h = lstm_out.shape
        x = lstm_out.view(-1,h)
        x = self.norm(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = F.softmax(x,dim=1)
        return x



if __name__ == '__main__':
    # 分割数据集，形成每个任务的数据集文件
    split_tasks_datasets(task_num)
    # 把每个任务的数据集封装成Dataset对象
    tasks_dataset = [Traffic_Task_Dataset(i) for i in range(task_num)]

    for i in range(task_num):
        print("the length of task{}'s dataset is {}".format(i,tasks_dataset[i].n_samples))


    # train_dataset,val_dataset = random_split(train_data,[math.floor(len(train_data)*0.7),len(train_data)-math.floor(len(train_data)*0.7)])
    with torch.backends.cudnn.flags(enabled=False):
        meta_model = LSTM_RNN().to(device='cuda')
        meta_optimizer = optim.Adam(meta_model.parameters(), beta, weight_decay=0.01)

        # initialize the alpha
        alpha = [torch.rand(para.size()).to('cuda') for para in meta_model.parameters()]
        alpha_optimizer = optim.Adam(alpha, 0.1, weight_decay=0.01)




        # train every task
        for epoch in range(5):
            print("epoch:{}".format(epoch))
            for task_dataset,i in zip(tasks_dataset,range(len(tasks_dataset)-num_test_task)):
                model = copy.deepcopy(meta_model)

                train_dataset, test_dataset = train_test_split(task_dataset, test_size=0.3, random_state=0,stratify=task_dataset.y_data, shuffle=True)

                # make the number of classes two:normal,abnormal
                for example in train_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    example = tuple(lst)
                for example in test_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    example = tuple(lst)


                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)

                # train
                criterion = nn.CrossEntropyLoss()
                # task_optimizer = optim.SGD(model.parameters(), lr=alpha)

                model.train()
                train_loss = 0
                sita_seccond_order_gradlist = []
                sita_first_order_gradlist = []
                post_sita_order = []

                for inputs, labels in train_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    for para in model.parameters():para.grad = None

                    outputs = model(inputs)
                    labels = labels.to(torch.int64)
                    labels = labels.view(-1)

                    loss = criterion(outputs, (F.one_hot(labels, num_classes=class_num)).to(torch.float64))

                    # get the seccond-order gradient and first-order gradient
                    dydx = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
                    d2ydx2 = [torch.autograd.grad(first_grad,para,retain_graph=True,grad_outputs=torch.ones_like(first_grad))[0] for first_grad,para in zip(dydx,model.parameters())]
                    sita_seccond_order_gradlist.append(d2ydx2)
                    sita_first_order_gradlist.append(dydx)

                    loss.backward()

                    sgd_optimize(model.parameters(), alpha, [para.grad for para in model.parameters()])

                    train_loss += loss.item()


                print('Train_task:{}, Loss per batch:{}'.format(i, train_loss / len(train_loader)))

                # test
                TP = 0.0
                FP = 0.0
                TN = 0.0
                FN = 0.0
                test_loss = 0

                for inputs, labels in test_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    outputs = model(inputs)
                    labels = labels.to(torch.int64)
                    labels = labels.view(-1)

                    loss = criterion(outputs, (F.one_hot(labels, num_classes=class_num)).to(torch.float64))
                    test_loss += loss

                    res = torch.argmax(outputs, dim=1)

                    for pre, truth in zip(res, labels):
                        if pre.item() == 1:
                            if truth.item() == 1:
                                TP += 1
                            else:
                                FP += 1

                        if pre.item() == 0:
                            if truth.item() == 0:
                                TN += 1
                            else:
                                FN += 1

                    # get the post_sita_order
                    post_sita_order = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

                print("Test_task:{}, Loss per batch:{}".format(i, test_loss.item() / len(test_loader)))
                print('accuracy:{},recall:{}'.format((TP + TN) / (TP + FP + TN + FN), TP / (TP + FN)))
                # compute the gradient of initial sita

                sum_seccond_gradient = [torch.zeros_like(x) for x in sita_seccond_order_gradlist[0]]
                for seccond_order_grad in sita_seccond_order_gradlist:
                    for gradsum, grad in zip(sum_seccond_gradient, seccond_order_grad):
                        gradsum.data += grad.data

                factor_rightlist = [1 - al * x for al,x in zip(alpha,sum_seccond_gradient)]

                ini_sita_grad = [ fac1*fac2 for fac1,fac2 in zip(post_sita_order,factor_rightlist)]

                # update the initial sita
                for ini_sita,ini_sita_grad in zip(meta_model.parameters(), ini_sita_grad):
                    ini_sita.grad = ini_sita_grad

                meta_optimizer.step()

                # compute the gradient of initial alpha
                sum_sita_first_gradient = None
                sum_sita_first_gradient = [torch.zeros_like(x) for x in sita_first_order_gradlist[0]]
                for first_order_grad in sita_first_order_gradlist:
                    for gradsum, grad in zip(sum_sita_first_gradient, first_order_grad):
                        gradsum.data += grad.data

                for  al,po,firstg in zip(alpha,post_sita_order,sum_sita_first_gradient):
                    al.grad = po * (-firstg)

                # update the initial alpha
                alpha_optimizer.step()

                print('--------------------------------------------')
            print('--------------------------------------------')
            print('test:')
            for task_dataset,i in zip(tasks_dataset[len(tasks_dataset)-num_test_task:len(tasks_dataset)],range(len(tasks_dataset)-num_test_task,len(tasks_dataset))):
                model = copy.deepcopy(meta_model)

                train_dataset, test_dataset = train_test_split(task_dataset, test_size=0.3, random_state=0,
                                                               stratify=task_dataset.y_data, shuffle=True)

                # make the number of classes two:normal,abnormal
                for example in train_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    example = tuple(lst)
                for example in test_dataset:
                    lst = list(example)
                    lst[1][0] = 1.0 if example[1][0] > 0 else 0.0
                    example = tuple(lst)

                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)

                # train
                criterion = nn.CrossEntropyLoss()
                # task_optimizer = optim.SGD(model.parameters(), lr=alpha)

                model.train()
                train_loss = 0
                sita_seccond_order_gradlist = []
                sita_first_order_gradlist = []
                post_sita_order = []

                for inputs, labels in train_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    for para in model.parameters(): para.grad = None

                    outputs = model(inputs)
                    labels = labels.to(torch.int64)
                    labels = labels.view(-1)

                    loss = criterion(outputs, (F.one_hot(labels, num_classes=class_num)).to(torch.float64))

                    # get the seccond-order gradient and first-order gradient
                    dydx = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
                    d2ydx2 = [
                        torch.autograd.grad(first_grad, para, retain_graph=True, grad_outputs=torch.ones_like(first_grad))[
                            0] for first_grad, para in zip(dydx, model.parameters())]
                    sita_seccond_order_gradlist.append(d2ydx2)
                    sita_first_order_gradlist.append(dydx)

                    loss.backward()

                    sgd_optimize(model.parameters(), alpha, [para.grad for para in model.parameters()])

                    train_loss += loss.item()

                print('Train_task:{}, Loss per batch:{}'.format(i, train_loss / len(train_loader)))

                # test
                TP = 0.0
                FP = 0.0
                TN = 0.0
                FN = 0.0
                test_loss = 0

                for inputs, labels in test_loader:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    outputs = model(inputs)
                    labels = labels.to(torch.int64)
                    labels = labels.view(-1)

                    loss = criterion(outputs, (F.one_hot(labels, num_classes=class_num)).to(torch.float64))
                    test_loss += loss

                    res = torch.argmax(outputs, dim=1)

                    for pre, truth in zip(res, labels):
                        if pre.item() == 1:
                            if truth.item() == 1:
                                TP += 1
                            else:
                                FP += 1

                        if pre.item() == 0:
                            if truth.item() == 0:
                                TN += 1
                            else:
                                FN += 1

                    # get the post_sita_order
                    post_sita_order = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)

                print("Test_task:{}, Loss per batch:{}".format(i, test_loss.item() / len(test_loader)))
                print('accuracy:{},recall:{}'.format((TP + TN) / (TP + FP + TN + FN), TP / (TP + FN)))
                # compute the gradient of initial sita

                # sum_seccond_gradient = [torch.zeros_like(x) for x in sita_seccond_order_gradlist[0]]
                # for seccond_order_grad in sita_seccond_order_gradlist:
                #     for gradsum, grad in zip(sum_seccond_gradient, seccond_order_grad):
                #         gradsum.data += grad.data
                #
                # factor_rightlist = [1 - al * x for al, x in zip(alpha, sum_seccond_gradient)]
                #
                # ini_sita_grad = [fac1 * fac2 for fac1, fac2 in zip(post_sita_order, factor_rightlist)]
                #
                # # update the initial sita
                # for ini_sita, ini_sita_grad in zip(meta_model.parameters(), ini_sita_grad):
                #     ini_sita.grad = ini_sita_grad
                #
                # meta_optimizer.step()
                #
                # # compute the gradient of initial alpha
                # sum_sita_first_gradient = None
                # sum_sita_first_gradient = [torch.zeros_like(x) for x in sita_first_order_gradlist[0]]
                # for first_order_grad in sita_first_order_gradlist:
                #     for gradsum, grad in zip(sum_sita_first_gradient, first_order_grad):
                #         gradsum.data += grad.data
                #
                # for al, po, firstg in zip(alpha, post_sita_order, sum_sita_first_gradient):
                #     al.grad = po * (-firstg)
                #
                # # update the initial alpha
                # alpha_optimizer.step()

            print('--------------------------------------------')
            print('--------------------------------------------')
            print('--------------------------------------------')