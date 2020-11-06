# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     main
   Description :
   Author :       walnut
   date:          2020/10/27
-------------------------------------------------
   Change Activity:
                  2020/10/27:
-------------------------------------------------
"""
__author__ = 'walnut'


from paras import *
from Utils.Reader import read_csv, read_excel_by_col
from Data.MyDataset import TransImagesSet
from torch.utils.data import DataLoader
from Models.CNN import CNN
import numpy as np
import re
import torch


# convert date format like '2019/5/1 16:05:15' in String
def date_transform(date):
    [year, month, day, hour, minute, second] = list(map(int, re.split("/| |:", date)))
    return [year, month, day, hour, minute, second]


def generate_trans_dict(trans_data, ids):
    trans_info_dicts = dict.fromkeys(ids, np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)))

    for item in trans_data:
        car_id = item[0]
        if car_id not in trans_info_dicts or len(item) < 10:
            continue

        car_type = int(item[1])
        ssid_in = int(item[2][3:])
        cdbh_in = int(item[3])
        date = date_transform(item[4])
        ssid_out = int(item[5][3:])
        cdbh_out = int(item[6])
        travel_time = int(item[8])
        date_type = int(item[9])

        trans_vector = np.array([car_type, date_type, ssid_in, cdbh_in, ssid_out, cdbh_out, travel_time])

        # ***********************************************************************
        # TO DO ...
        # This need to be update according to selected length of trans period
        # Here, take one month and one hour as example
        trans_info_dicts[car_id][date[2]][date[4]] = trans_vector
        # ***********************************************************************

    return trans_info_dicts


def train(epoch, my_model, data_loader, optimizer, loss_func, device):
    model.train()

    # record training data
    ground_truth = []
    prediction = []

    for train_step, (train_image, train_label, train_label_id) in enumerate(data_loader):
        train_image = train_image.to(device)
        train_label_id = train_label_id.type(torch.FloatTensor).to(device)

        train_output = model(train_image)
        # *********************************************************************
        # loss function need to update
        # loss = loss_func(output, label_id)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        ground_truth.extend(train_label_id.cpu().detach().numpy().tolist())
        prediction.extend(train_output.cpu().detach().numpy().tolist())

    # ******************************** TO DO*******************************
        print("\t{}-{}\t\tTrain_loss: {:.4f}".format(epoch, step, ))
    print("\tAverage train loss: {:.4f}".format())
    print("\tAverage train error:{:.4f}\n".format())

    return prediction


if __name__ == "__main__":

    # file read test
    # print(len(read_csv(DATA_FILE_PATH + DATA_FILE_NAME)[0:10]))
    # print(read_csv_by_row(DATA_FILE_PATH+DATA_FILE_NAME, 3))
    # print(read_csv_by_col(DATA_FILE_PATH+DATA_FILE_NAME, "HPHM")[0:10])
    # print(read_excel_by_col(FILE_PATH + CAR_IDS_FILE_NAME)[0])

    csv_data = read_csv(FILE_PATH + DATA_FILE_NAME)

    # read non-repetitive car ID numbers into a list
    car_ids = read_excel_by_col(FILE_PATH + CAR_IDS_FILE_NAME)[0]

    # generate dict [car_id->trans_image]
    trans_image_dict = generate_trans_dict(csv_data, car_ids)

    print("统计车牌总数：%d\n" % len(trans_image_dict))

    # data print test (last case)
    # np.set_printoptions(suppress=True)
    # case = trans_image_dict.popitem()
    # print("车牌号：%s" % case[0])
    # print(np.array(case[1]))


    # device configuration
    device = torch.device("cuda:" + GPU_ID if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True

    train_data = TransImagesSet(trans_dicts=trans_image_dict)
    train_loader = DataLoader(dataset=train_data, shuffle=False, batch_size=BATCH_TRAIN, num_workers=4)

    model = CNN()
    model.to(device)
    print(model)

    # ************************* model training ******************************************
    # Need to update ...
    # my_optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=2e-4)

    # for epoch in range(1, EPOCH + 1):
    #     print("Epoch {}: (with LR = {}):".format(epoch, my_optimizer.param_groups[0]['lr']))
    #     train(epoch=epoch, model=model, data_loader=train_loader,
    #                        optimizer=my_optimizer, loss_func=torch.nn.MSELoss, device=device)

    ids = []
    outputs = []      # model output from trans images

    for step, (image, label, label_id) in enumerate(train_loader):
        image = image.type(torch.FloatTensor).to(device)
        output = model(image)
        outputs.extend(output.cpu().detach().numpy().tolist())
        ids.extend(label)

    print("车牌号(总数{})：{}".format(len(ids), ids))
    print("特征向量(总数%d)：%s" % (len(outputs), outputs))

    pass


