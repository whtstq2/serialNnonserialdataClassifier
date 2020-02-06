import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import lightgbm as lgb
import sys, os, glob
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')

# matrix print all option
# np.set_printoptions(threshold= sys.maxsize)


def split_to_train_test(df, label_column, train_frac):
    print('---------------------------------------------------------------------------')
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        print('\n%s:\n--------\ntotal:%d\ntrain_df:%d\ntest_df:%d' % (lbl, len(lbl_df), len(lbl_train_df), len(lbl_test_df)))
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)

    return train_df, test_df

# train
def train_with_xgboost(train_data, train_label, test_data, test_label):
    model = XGBClassifier(min_child_weight=1, max_depth=5, gamma=0, colsample_bytree=0.8,
                      objective='multi:softmax', eval_metric='rmse', eta=0.3, num_round=1000, max_leaves=15)

    model.fit(train_data, train_label.ravel())

    # make predictions
    expected_y = test_label
    predicted_y = model.predict(test_data)

    # summarize the fit of the model
    print('XGBoost: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))
    print(confusion_matrix(train_label, model.predict(train_data)))



def train_with_lightGBM(train_data, train_label, test_data, test_label):
    model = lgb.LGBMClassifier(boosting_type='gbdt',
                               max_depth=-1,
                               colsample_bytree=1.0,
                               learning_rate=0.05,
                               n_estimators=100,
                               subsample_for_bin=200000,
                               objective='multiclass',
                               class_weight=None,
                               min_split_gain=0.,
                               min_child_weight=1e-3,
                               min_child_samples=5,
                               subsample=1.,
                               subsample_freq=0,
                               reg_alpha=0.,
                               reg_lambda=0.,
                               random_state=None,
                               n_jobs=-1,
                               silent=True,
                               importance_type='split')

    model.fit(train_data, train_label.ravel())

    # make predictions
    expected_y = test_label
    predicted_y = model.predict(test_data)

    # summarize the fit of the model
    print('LightGBM: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))
    print(confusion_matrix(train_label, model.predict(train_data)))
    print('-------------------------------------------------------------------')

######################################################################################################################


data_path = 'C:/Users/whtstq2/Desktop/smartGolf/data/rawdata'


list_of_files = glob.glob(os.path.join(data_path, '*'))


all_data_frames = []
all_data_labels = []


for file in list_of_files:
    data_label = [file[73], file[74], file[75], file[76], file[77], file[78], file[79], file[80], file[81], file[82]]
    data_label = np.array(data_label)
    all_data_labels = np.concatenate((all_data_labels, data_label), axis=0)

all_data_labels = all_data_labels.reshape(len(list_of_files), 10)


for file in list_of_files:
    data_frame = pd.read_csv(file, index_col=None)

    # 데이터 인덱싱 for parameter selection
    data_frame = np.array(data_frame)
    data_frame1 = np.delete(data_frame, (6,7,8,9,10,11), axis=1)
    # 데이터 zero-padding
    data_frame1 = np.concatenate((data_frame1, np.zeros((1000 - len(data_frame1), 7), dtype=float)), axis=0)
    
    # xgboost에 넣기 위해 합침
    all_data_frames.append(data_frame1)



data_frame_stack = np.stack(all_data_frames, axis=2)
# 2차원으로 reshape
data_frame_stack = data_frame_stack.reshape(376,7000)

# concat data and label
# all_data_labels = pd.DataFrame(all_data_labels, columns=["grip", "address", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"])
gripData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, :1]), axis=1)
gripData_with_label = pd.DataFrame(gripData_with_label)
gripData_with_label.rename(columns={gripData_with_label.columns[7000]:"grip"}, inplace=True)

# ratio split
dataGripTrainset, dataGripTestset = split_to_train_test(gripData_with_label, 'grip', 0.9)

# convert dataframe to numpy
dataGripTrainset = dataGripTrainset.to_numpy()
dataGripTestset = dataGripTestset.to_numpy()

# train, testset labeling
dataGripTrainD, dataGripTrainL = dataGripTrainset[:, :7000], dataGripTrainset[:, 7000:]
dataGripTestD, dataGripTestL = dataGripTestset[:, :7000], dataGripTestset[:, 7000:]


# train data standardization
scaler = StandardScaler()
dataGripTrainD_scaled = scaler.fit_transform(dataGripTrainD)

# train
train_with_xgboost(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_lightGBM(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)

######## ADDRESS LABEL ##############
addressData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 1:2]), axis=1)
addressData_with_label = pd.DataFrame(addressData_with_label)
addressData_with_label.rename(columns={addressData_with_label.columns[7000]:"address"}, inplace=True)

dataAddressTrainset, dataAddressTestset = split_to_train_test(addressData_with_label, 'address', 0.9)
dataAddressTrainset = dataAddressTrainset.to_numpy()
dataAddressTestset = dataAddressTestset.to_numpy()
dataAddressTrainD, dataAddressTrainL = dataAddressTrainset[:, :7000], dataAddressTrainset[:, 7000:]
dataAddressTestD, dataAddressTestL = dataAddressTestset[:, :7000], dataAddressTestset[:, 7000:]
scaler = StandardScaler()
dataAddressTrainD_scaled = scaler.fit_transform(dataAddressTrainD)
train_with_xgboost(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_lightGBM(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)

######## BACKCOCK LABEL ##############
backcockData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 2:3]), axis=1)
backcockData_with_label = pd.DataFrame(backcockData_with_label)
backcockData_with_label.rename(columns={backcockData_with_label.columns[7000]:"address"}, inplace=True)

dataBackcockTrainset, dataBackcockTestset = split_to_train_test(backcockData_with_label, 'address', 0.9)
dataBackcockTrainset = dataBackcockTrainset.to_numpy()
dataBackcockTestset = dataBackcockTestset.to_numpy()
dataBackcockTrainD, dataBackcockTrainL = dataBackcockTrainset[:, :7000], dataBackcockTrainset[:, 7000:]
dataBackcockTestD, dataBackcockTestL = dataBackcockTestset[:, :7000], dataBackcockTestset[:, 7000:]
scaler = StandardScaler()
dataBackcockTrainD_scaled = scaler.fit_transform(dataBackcockTrainD)
train_with_xgboost(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_lightGBM(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)



######## TOP LABEL ##############
topData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 3:4]), axis=1)
topData_with_label = pd.DataFrame(topData_with_label)
topData_with_label.rename(columns={topData_with_label.columns[7000]:"top"}, inplace=True)
dataTopTrainset, dataTopTestset = split_to_train_test(topData_with_label, 'top', 0.9)
dataTopTrainset = dataTopTrainset.to_numpy()
dataTopTestset = dataTopTestset.to_numpy()
dataTopTrainD, dataTopTrainL = dataTopTrainset[:, :7000], dataTopTrainset[:, 7000:]
dataTopTestD, dataTopTestL = dataTopTestset[:, :7000], dataTopTestset[:, 7000:]
scaler = StandardScaler()
dataTopTrainD_scaled = scaler.fit_transform(dataTopTrainD)
train_with_xgboost(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_lightGBM(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)


######## DOWNBODY LABEL ##############
downbodyData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 4:5]), axis=1)
downbodyData_with_label = pd.DataFrame(downbodyData_with_label)
downbodyData_with_label.rename(columns={downbodyData_with_label.columns[7000]:"downBody"}, inplace=True)
dataDownbodyTrainset, dataDownbodyTestset = split_to_train_test(downbodyData_with_label, 'downBody', 0.9)
dataDownbodyTrainset = dataDownbodyTrainset.to_numpy()
dataDownbodyTestset = dataDownbodyTestset.to_numpy()
dataDownbodyTrainD, dataDownbodyTrainL = dataDownbodyTrainset[:, :7000], dataDownbodyTrainset[:, 7000:]
dataDownbodyTestD, dataDownbodyTestL = dataDownbodyTestset[:, :7000], dataDownbodyTestset[:, 7000:]
scaler = StandardScaler()
dataDownbodyTrainD_scaled = scaler.fit_transform(dataDownbodyTrainD)
train_with_xgboost(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_lightGBM(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)



######## DOWNCOCK LABEL ##############
downcockData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 5:6]), axis=1)
downcockData_with_label = pd.DataFrame(downcockData_with_label)
downcockData_with_label.rename(columns={downcockData_with_label.columns[7000]:"downCock"}, inplace=True)
dataDowncockTrainset, dataDowncockTestset = split_to_train_test(downcockData_with_label, 'downCock', 0.9)
dataDowncockTrainset = dataDowncockTrainset.to_numpy()
dataDowncockTestset = dataDowncockTestset.to_numpy()
dataDowncockTrainD, dataDowncockTrainL = dataDowncockTrainset[:, :7000], dataDowncockTrainset[:, 7000:]
dataDowncockTestD, dataDowncockTestL = dataDowncockTestset[:, :7000], dataDowncockTestset[:, 7000:]
scaler = StandardScaler()
dataDowncockTrainD_scaled = scaler.fit_transform(dataDowncockTrainD)
train_with_xgboost(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_lightGBM(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)



######## IMPACTFACE LABEL ##############
impactfaceData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 6:7]), axis=1)
impactfaceData_with_label = pd.DataFrame(impactfaceData_with_label)
impactfaceData_with_label.rename(columns={impactfaceData_with_label.columns[7000]:"impactFace"}, inplace=True)
dataImpactfaceTrainset, dataImpactfaceTestset = split_to_train_test(impactfaceData_with_label, 'impactFace', 0.9)
dataImpactfaceTrainset = dataImpactfaceTrainset.to_numpy()
dataImpactfaceTestset = dataImpactfaceTestset.to_numpy()
dataImpactfaceTrainD, dataImpactfaceTrainL = dataImpactfaceTrainset[:, :7000], dataImpactfaceTrainset[:, 7000:]
dataImpactfaceTestD, dataImpactfaceTestL = dataImpactfaceTestset[:, :7000], dataImpactfaceTestset[:, 7000:]
scaler = StandardScaler()
dataImpactfaceTrainD_scaled = scaler.fit_transform(dataImpactfaceTrainD)
train_with_xgboost(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_lightGBM(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)



######## IMPACTPASS LABEL ##############
impactpassData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 7:8]), axis=1)
impactpassData_with_label = pd.DataFrame(impactpassData_with_label)
impactpassData_with_label.rename(columns={impactpassData_with_label.columns[7000]:"impactPass"}, inplace=True)
dataImpactpassTrainset, dataImpactpassTestset = split_to_train_test(impactpassData_with_label, 'impactPass', 0.9)
dataImpactpassTrainset = dataImpactpassTrainset.to_numpy()
dataImpactpassTestset = dataImpactpassTestset.to_numpy()
dataImpactpassTrainD, dataImpactpassTrainL = dataImpactpassTrainset[:, :7000], dataImpactpassTrainset[:, 7000:]
dataImpactpassTestD, dataImpactpassTestL = dataImpactpassTestset[:, :7000], dataImpactpassTestset[:, 7000:]
scaler = StandardScaler()
dataImpactpassTrainD_scaled = scaler.fit_transform(dataImpactpassTrainD)
train_with_xgboost(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_lightGBM(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)



######## FOLLOWSWING LABEL ##############
followswingData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 8:9]), axis=1)
followswingData_with_label = pd.DataFrame(followswingData_with_label)
followswingData_with_label.rename(columns={followswingData_with_label.columns[7000]:"followSwing"}, inplace=True)
dataFollowswingTrainset, dataFollowswingTestset = split_to_train_test(followswingData_with_label, 'followSwing', 0.9)
dataFollowswingTrainset = dataFollowswingTrainset.to_numpy()
dataFollowswingTestset = dataFollowswingTestset.to_numpy()
dataFollowswingTrainD, dataFollowswingTrainL = dataFollowswingTrainset[:, :7000], dataFollowswingTrainset[:, 7000:]
dataFollowswingTestD, dataFollowswingTestL = dataFollowswingTestset[:, :7000], dataFollowswingTestset[:, 7000:]
scaler = StandardScaler()
dataFollowswingTrainD_scaled = scaler.fit_transform(dataFollowswingTrainD)
train_with_xgboost(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_lightGBM(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)



######## FOLLOWTOP LABEL ##############
followtopData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 9:10]), axis=1)
followtopData_with_label = pd.DataFrame(followtopData_with_label)
followtopData_with_label.rename(columns={followtopData_with_label.columns[7000]:"followTop"}, inplace=True)
dataFollowtopTrainset, dataFollowtopTestset = split_to_train_test(followtopData_with_label, 'followTop', 0.9)
dataFollowtopTrainset = dataFollowtopTrainset.to_numpy()
dataFollowtopTestset = dataFollowtopTestset.to_numpy()
dataFollowtopTrainD, dataFollowtopTrainL = dataFollowtopTrainset[:, :7000], dataFollowtopTrainset[:, 7000:]
dataFollowtopTestD, dataFollowtopTestL = dataFollowtopTestset[:, :7000], dataFollowtopTestset[:, 7000:]
scaler = StandardScaler()
dataFollowtopTrainD_scaled = scaler.fit_transform(dataFollowtopTrainD)
train_with_xgboost(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_lightGBM(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
