import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
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
    print('-------------------------------------------------------------------')



######################################################################################################################



data_path = 'C:/Users/whtstq2/Desktop/smartGolf/data/data2446.xlsx'

df = pd.read_excel(data_path, dtype={"Type": np.int64, "Sex": np.int64, "Height": np.int64, "Weight":np.int64,
                                     "Hand": np.int64, "Club": np.int64, "GameClub": np.int64, "Speed": float,
                                     "FaceAngle": float, "PathAngle": float,
                                     "AttackAngle": float, "SpinLoft": float, "SmashFactor": float, "Tempo": float,
                                     "Ball": str, "AddressClubX": float,   "AddressClubZ": float, "ImpactClubX": float,
                                     "ImpactClubZ": float, "TopTime": float, "ImpactTime": float, "FinishTime": float,
                                     "TotalTime": float, "ArraySize": np.int64, "saveTime":str, "id": np.int64,
                                     "grip": np.int64, "address": np.int64, "backcock": np.int64, "top": np.int64,
                                     "downBody": np.int64, "downCock": np.int64, "impactFace": np.int64,
                                     "impactPass": np.int64, "followSwing": np.int64, "followTop": np.int64})


df = df.drop(['Ball', "saveTime"], axis=1)



######## GRIP LABEL ##############
# one label data
dataGrip = df.drop(["address", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)

# data ratio split
dataGripTrainset, dataGripTestset = split_to_train_test(dataGrip, 'grip', 0.9)
print(np.shape(dataGripTrainset), np.shape(dataGripTestset))

# convert dataframe to numpy
dataGripTrainset = dataGripTrainset.to_numpy()
dataGripTestset = dataGripTestset.to_numpy()

# train, testset label slicing
dataGripTrainD, dataGripTrainL = dataGripTrainset[:, :24], dataGripTrainset[:, 24:]
print(np.shape(dataGripTrainD), np.shape(dataGripTrainL))
dataGripTestD, dataGripTestL = dataGripTestset[:, :24], dataGripTestset[:, 24:]
print(np.shape(dataGripTestD), np.shape(dataGripTestL))


# train data standardization
scaler = StandardScaler()
dataGripTrainD_scaled = scaler.fit_transform(dataGripTrainD)

model = XGBClassifier()
model.fit(dataGripTrainD, dataGripTrainL)

#
model = XGBClassifier()
model.fit(dataGripTrainD, dataGripTrainL)
#feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()