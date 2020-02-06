import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import lightgbm as lgb
import sys, os, glob
import warnings

warnings.filterwarnings('ignore')

# matrix print all option
np.set_printoptions(threshold= sys.maxsize)


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
                      objective='multi:softmax', eval_metric='rmse', eta=0.3, num_round=1000, max_leaves=15, num_class=3)

    model.fit(train_data, train_label.ravel())

    # make predictions
    expected_y = test_label
    predicted_y = model.predict(test_data)

    # summarize the fit of the model
    print('XGBoost: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))



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
                               importance_type='split',
                               num_class=3)

    model.fit(train_data, train_label.ravel())

    # make predictions
    expected_y = test_label
    predicted_y = model.predict(test_data)

    # summarize the fit of the model
    print('LightGBM: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))
    print('-------------------------------------------------------------------')


def train_with_svm(train_data, train_label, test_data, test_label):
    model = svm.SVC(kernel='linear')
    model.fit(train_data, train_label)
    expected_y = test_label
    predicted_y = model.predict(test_data)

    print('SVM: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))


def train_with_mlp(train_data, train_label, test_data, test_label):
    model = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=1000).fit(train_data, train_label)
    expected_y = test_label
    predicted_y = model.predict(test_data)

    print('MLP: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))


def train_with_rf(train_data, train_label, test_data, test_label):
    model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=906419)
    model.fit(train_data, train_label)
    expected_y = test_label
    predicted_y = model.predict(test_data)
    accuracy = accuracy_score(expected_y, predicted_y)
    print('RF: ')
    print(f'Out-of-bag score estimate: {model.oob_score_:.3}')
    print(f'Mean accuracy score: {accuracy:.3}')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))

######################################################################################################################

print(device_lib.list_local_devices())


######################################################################################################################
data_path = 'C:/Users/whtstq2/Desktop/smartGolf/raw/data/data'


list_of_files = glob.glob(os.path.join(data_path, '*'))


all_data_frames = []
all_data_labels = []


for file in list_of_files:
    data_label = [file[74], file[75], file[76],file[77], file[78], file[79], file[80], file[81], file[82], file[83]]
    data_label = np.array(data_label)
    all_data_labels = np.concatenate((all_data_labels, data_label), axis=0)

all_data_labels = all_data_labels.reshape(len(list_of_files), 10)


for file in list_of_files:
    data_frame = pd.read_csv(file, index_col=None)

    # data indexing for parameter selection
    data_frame = np.array(data_frame)

    # data zero-padding
    data_frame = np.concatenate((data_frame, np.zeros((1000 - len(data_frame), 13), dtype=float)), axis=0)
    
    # xgboost에 넣기 위해 합침
    all_data_frames.append(data_frame)



data_frame_stack = np.stack(all_data_frames, axis=2)
# 2차원으로 reshape
data_frame_stack = data_frame_stack.reshape(385,13000)


'''
# train data standardization
scaler = StandardScaler()
dataGripTrainD_scaled = scaler.fit_transform(dataGripTrainD)
'''

'''
# 샘플 데이터 표현
plt.scatter(dataGripTrainD[:, 0], dataGripTrainD[:, 1], c=dataGripTrainL/16, s=30, cmap=plt.cm.Paired)

# 초평면(Hyper-Plane) 표현
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 지지벡터(Support Vector) 표현
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=60, facecolors='r')

plt.show()

'''

############ GRIP ################################
# concat data and label
# all_data_labels = pd.DataFrame(all_data_labels, columns=["grip", "address", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"])
gripData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, :1]), axis=1)
gripData_with_label = pd.DataFrame(gripData_with_label)
gripData_with_label.rename(columns={gripData_with_label.columns[13000]:"grip"}, inplace=True)

# ratio split
dataGripTrainset, dataGripTestset = split_to_train_test(gripData_with_label, 'grip', 0.8)

# convert dataframe to numpy
dataGripTrainset = dataGripTrainset.to_numpy()
dataGripTestset = dataGripTestset.to_numpy()

# train, testset labeling
dataGripTrainD, dataGripTrainL = dataGripTrainset[:, :13000], dataGripTrainset[:, 13000:]
dataGripTestD, dataGripTestL = dataGripTestset[:, :13000], dataGripTestset[:, 13000:]


# train data standardization
scaler = StandardScaler()
dataGripTrainD_scaled = scaler.fit_transform(dataGripTrainD)

# train
print('[grip]')
train_with_xgboost(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_lightGBM(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_svm(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_mlp(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_rf(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)

######## ADDRESS LABEL ##############
addressData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 1:2]), axis=1)
addressData_with_label = pd.DataFrame(addressData_with_label)
addressData_with_label.rename(columns={addressData_with_label.columns[13000]:"address"}, inplace=True)

dataAddressTrainset, dataAddressTestset = split_to_train_test(addressData_with_label, 'address', 0.8)
dataAddressTrainset = dataAddressTrainset.to_numpy()
dataAddressTestset = dataAddressTestset.to_numpy()
dataAddressTrainD, dataAddressTrainL = dataAddressTrainset[:, :13000], dataAddressTrainset[:, 13000:]
dataAddressTestD, dataAddressTestL = dataAddressTestset[:, :13000], dataAddressTestset[:, 13000:]
scaler = StandardScaler()
dataAddressTrainD_scaled = scaler.fit_transform(dataAddressTrainD)
print('[address]')
train_with_xgboost(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_lightGBM(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_svm(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_mlp(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_rf(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)


######## BACKCOCK LABEL ##############
backcockData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 2:3]), axis=1)
backcockData_with_label = pd.DataFrame(backcockData_with_label)
backcockData_with_label.rename(columns={backcockData_with_label.columns[13000]:"address"}, inplace=True)

dataBackcockTrainset, dataBackcockTestset = split_to_train_test(backcockData_with_label, 'address', 0.8)
dataBackcockTrainset = dataBackcockTrainset.to_numpy()
dataBackcockTestset = dataBackcockTestset.to_numpy()
dataBackcockTrainD, dataBackcockTrainL = dataBackcockTrainset[:, :13000], dataBackcockTrainset[:, 13000:]
dataBackcockTestD, dataBackcockTestL = dataBackcockTestset[:, :13000], dataBackcockTestset[:, 13000:]
scaler = StandardScaler()
dataBackcockTrainD_scaled = scaler.fit_transform(dataBackcockTrainD)
print('[backcock]')
train_with_xgboost(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_lightGBM(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_svm(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_mlp(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_rf(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)


######## TOP LABEL ##############
topData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 3:4]), axis=1)
topData_with_label = pd.DataFrame(topData_with_label)
topData_with_label.rename(columns={topData_with_label.columns[13000]:"top"}, inplace=True)
dataTopTrainset, dataTopTestset = split_to_train_test(topData_with_label, 'top', 0.8)
dataTopTrainset = dataTopTrainset.to_numpy()
dataTopTestset = dataTopTestset.to_numpy()
dataTopTrainD, dataTopTrainL = dataTopTrainset[:, :13000], dataTopTrainset[:, 13000:]
dataTopTestD, dataTopTestL = dataTopTestset[:, :13000], dataTopTestset[:, 13000:]
scaler = StandardScaler()
dataTopTrainD_scaled = scaler.fit_transform(dataTopTrainD)
print('[top]')
train_with_xgboost(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_lightGBM(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_svm(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_mlp(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_rf(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)


######## DOWNBODY LABEL ##############
downbodyData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 4:5]), axis=1)
downbodyData_with_label = pd.DataFrame(downbodyData_with_label)
downbodyData_with_label.rename(columns={downbodyData_with_label.columns[13000]:"downBody"}, inplace=True)
dataDownbodyTrainset, dataDownbodyTestset = split_to_train_test(downbodyData_with_label, 'downBody', 0.8)
dataDownbodyTrainset = dataDownbodyTrainset.to_numpy()
dataDownbodyTestset = dataDownbodyTestset.to_numpy()
dataDownbodyTrainD, dataDownbodyTrainL = dataDownbodyTrainset[:, :13000], dataDownbodyTrainset[:, 13000:]
dataDownbodyTestD, dataDownbodyTestL = dataDownbodyTestset[:, :13000], dataDownbodyTestset[:, 13000:]
scaler = StandardScaler()
dataDownbodyTrainD_scaled = scaler.fit_transform(dataDownbodyTrainD)
print('[downbody]')
train_with_xgboost(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_lightGBM(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_svm(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_mlp(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_rf(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)


######## DOWNCOCK LABEL ##############
downcockData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 5:6]), axis=1)
downcockData_with_label = pd.DataFrame(downcockData_with_label)
downcockData_with_label.rename(columns={downcockData_with_label.columns[13000]:"downCock"}, inplace=True)
dataDowncockTrainset, dataDowncockTestset = split_to_train_test(downcockData_with_label, 'downCock', 0.8)
dataDowncockTrainset = dataDowncockTrainset.to_numpy()
dataDowncockTestset = dataDowncockTestset.to_numpy()
dataDowncockTrainD, dataDowncockTrainL = dataDowncockTrainset[:, :13000], dataDowncockTrainset[:, 13000:]
dataDowncockTestD, dataDowncockTestL = dataDowncockTestset[:, :13000], dataDowncockTestset[:, 13000:]
scaler = StandardScaler()
dataDowncockTrainD_scaled = scaler.fit_transform(dataDowncockTrainD)
print('[downcock]')
train_with_xgboost(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_lightGBM(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_svm(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_mlp(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_rf(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)


######## IMPACTFACE LABEL ##############
impactfaceData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 6:7]), axis=1)
impactfaceData_with_label = pd.DataFrame(impactfaceData_with_label)
impactfaceData_with_label.rename(columns={impactfaceData_with_label.columns[13000]:"impactFace"}, inplace=True)
dataImpactfaceTrainset, dataImpactfaceTestset = split_to_train_test(impactfaceData_with_label, 'impactFace', 0.8)
dataImpactfaceTrainset = dataImpactfaceTrainset.to_numpy()
dataImpactfaceTestset = dataImpactfaceTestset.to_numpy()
dataImpactfaceTrainD, dataImpactfaceTrainL = dataImpactfaceTrainset[:, :13000], dataImpactfaceTrainset[:, 13000:]
dataImpactfaceTestD, dataImpactfaceTestL = dataImpactfaceTestset[:, :13000], dataImpactfaceTestset[:, 13000:]
scaler = StandardScaler()
dataImpactfaceTrainD_scaled = scaler.fit_transform(dataImpactfaceTrainD)
print('[impactface]')
train_with_xgboost(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_lightGBM(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_svm(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_mlp(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_rf(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)


######## IMPACTPASS LABEL ##############
impactpassData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 7:8]), axis=1)
impactpassData_with_label = pd.DataFrame(impactpassData_with_label)
impactpassData_with_label.rename(columns={impactpassData_with_label.columns[13000]:"impactPass"}, inplace=True)
dataImpactpassTrainset, dataImpactpassTestset = split_to_train_test(impactpassData_with_label, 'impactPass', 0.8)
dataImpactpassTrainset = dataImpactpassTrainset.to_numpy()
dataImpactpassTestset = dataImpactpassTestset.to_numpy()
dataImpactpassTrainD, dataImpactpassTrainL = dataImpactpassTrainset[:, :13000], dataImpactpassTrainset[:, 13000:]
dataImpactpassTestD, dataImpactpassTestL = dataImpactpassTestset[:, :13000], dataImpactpassTestset[:, 13000:]
scaler = StandardScaler()
dataImpactpassTrainD_scaled = scaler.fit_transform(dataImpactpassTrainD)
print('[impactpass]')
train_with_xgboost(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_lightGBM(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_svm(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_mlp(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_rf(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)

######## FOLLOWSWING LABEL ##############
followswingData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 8:9]), axis=1)
followswingData_with_label = pd.DataFrame(followswingData_with_label)
followswingData_with_label.rename(columns={followswingData_with_label.columns[13000]:"followSwing"}, inplace=True)
dataFollowswingTrainset, dataFollowswingTestset = split_to_train_test(followswingData_with_label, 'followSwing', 0.8)
dataFollowswingTrainset = dataFollowswingTrainset.to_numpy()
dataFollowswingTestset = dataFollowswingTestset.to_numpy()
dataFollowswingTrainD, dataFollowswingTrainL = dataFollowswingTrainset[:, :13000], dataFollowswingTrainset[:, 13000:]
dataFollowswingTestD, dataFollowswingTestL = dataFollowswingTestset[:, :13000], dataFollowswingTestset[:, 13000:]
scaler = StandardScaler()
dataFollowswingTrainD_scaled = scaler.fit_transform(dataFollowswingTrainD)
print('[followswing]')
train_with_xgboost(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_lightGBM(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_svm(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_mlp(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_rf(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)


######## FOLLOWTOP LABEL ##############
followtopData_with_label = np.concatenate((data_frame_stack, all_data_labels[:, 9:10]), axis=1)
followtopData_with_label = pd.DataFrame(followtopData_with_label)
followtopData_with_label.rename(columns={followtopData_with_label.columns[13000]:"followTop"}, inplace=True)
dataFollowtopTrainset, dataFollowtopTestset = split_to_train_test(followtopData_with_label, 'followTop', 0.8)
dataFollowtopTrainset = dataFollowtopTrainset.to_numpy()
dataFollowtopTestset = dataFollowtopTestset.to_numpy()
dataFollowtopTrainD, dataFollowtopTrainL = dataFollowtopTrainset[:, :13000], dataFollowtopTrainset[:, 13000:]
dataFollowtopTestD, dataFollowtopTestL = dataFollowtopTestset[:, :13000], dataFollowtopTestset[:, 13000:]
scaler = StandardScaler()
dataFollowtopTrainD_scaled = scaler.fit_transform(dataFollowtopTrainD)
print('[followtop]')
train_with_xgboost(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_lightGBM(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_svm(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_mlp(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_rf(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)