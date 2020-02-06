
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
from xgboost import plot_importance
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


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
                      objective='multi:softmax', eval_metric='rmse', eta=0.05, num_round=1000, max_leaves=15)

    model.fit(train_data, train_label)

    # make predictions
    expected_y = test_label
    predicted_y = model.predict(test_data)

    # summarize the fit of the model
    print('XGBoost: ')
    print(classification_report(expected_y, predicted_y))
    print(confusion_matrix(expected_y, predicted_y))



def train_with_lightGBM(train_data, train_label, test_data, test_label):
    model = lgb.LGBMClassifier(boosting_type='gbdt',
                               feature_fraction=0.8,
                               metric='multi_logloss',
                               num_leaves=15,
                               max_depth=5,
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
                               colsample_bytree=1.,
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


df = df.drop(['Ball', 'saveTime'], axis=1)



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

# train
train_with_xgboost(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_lightGBM(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)

######## ADDRESS LABEL ##############
dataAddress = df.drop(["grip", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)
dataAddressTrainset, dataAddressTestset = split_to_train_test(dataAddress, 'address', 0.9)
dataAddressTrainset = dataAddressTrainset.to_numpy()
dataAddressTestset = dataAddressTestset.to_numpy()
dataAddressTrainD, dataAddressTrainL = dataAddressTrainset[:, :24], dataAddressTrainset[:, 24:]
dataAddressTestD, dataAddressTestL = dataAddressTestset[:, :24], dataAddressTestset[:, 24:]
scaler = StandardScaler()
dataAddressTrainD_scaled = scaler.fit_transform(dataAddressTrainD)
train_with_xgboost(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_lightGBM(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)

######## BACKCOCK LABEL ##############
dataBackcock = df.drop(["grip", "address", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)
dataBackcockTrainset, dataBackcockTestset = split_to_train_test(dataBackcock, 'backcock', 0.9)
dataBackcockTrainset = dataBackcockTrainset.to_numpy()
dataBackcockTestset = dataBackcockTestset.to_numpy()
dataBackcockTrainD, dataBackcockTrainL = dataBackcockTrainset[:, :24], dataBackcockTrainset[:, 24:]
dataBackcockTestD, dataBackcockTestL = dataBackcockTestset[:, :24], dataBackcockTestset[:, 24:]
scaler = StandardScaler()
dataBackcockTrainD_scaled = scaler.fit_transform(dataBackcockTrainD)
train_with_xgboost(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_lightGBM(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)



######## TOP LABEL ##############
dataTop = df.drop(["grip", "address", "backcock", "downBody", "downCock", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)
dataTopTrainset, dataTopTestset = split_to_train_test(dataTop, 'top', 0.9)
dataTopTrainset = dataTopTrainset.to_numpy()
dataTopTestset = dataTopTestset.to_numpy()
dataTopTrainD, dataTopTrainL = dataTopTrainset[:, :24], dataTopTrainset[:, 24:]
dataTopTestD, dataTopTestL = dataTopTestset[:, :24], dataTopTestset[:, 24:]
scaler = StandardScaler()
dataTopTrainD_scaled = scaler.fit_transform(dataTopTrainD)
train_with_xgboost(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_lightGBM(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)


######## DOWNBODY LABEL ##############
dataDownbody = df.drop(["grip", "address", "backcock", "top", "downCock", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)
dataDownbodyTrainset, dataDownbodyTestset = split_to_train_test(dataDownbody, 'downBody', 0.9)
dataDownbodyTrainset = dataDownbodyTrainset.to_numpy()
dataDownbodyTestset = dataDownbodyTestset.to_numpy()
dataDownbodyTrainD, dataDownbodyTrainL = dataDownbodyTrainset[:, :24], dataDownbodyTrainset[:, 24:]
dataDownbodyTestD, dataDownbodyTestL = dataDownbodyTestset[:, :24], dataDownbodyTestset[:, 24:]
scaler = StandardScaler()
dataDownbodyTrainD_scaled = scaler.fit_transform(dataDownbodyTrainD)
train_with_xgboost(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_lightGBM(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)



######## DOWNCOCK LABEL ##############
dataDowncock = df.drop(["grip", "address", "backcock", "top", "downBody", "impactFace", "impactPass", "followSwing", "followTop"], axis=1)
dataDowncockTrainset, dataDowncockTestset = split_to_train_test(dataDowncock, 'downCock', 0.9)
dataDowncockTrainset = dataDowncockTrainset.to_numpy()
dataDowncockTestset = dataDowncockTestset.to_numpy()
dataDowncockTrainD, dataDowncockTrainL = dataDowncockTrainset[:, :24], dataDowncockTrainset[:, 24:]
dataDowncockTestD, dataDowncockTestL = dataDowncockTestset[:, :24], dataDowncockTestset[:, 24:]
scaler = StandardScaler()
dataDowncockTrainD_scaled = scaler.fit_transform(dataDowncockTrainD)
train_with_xgboost(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_lightGBM(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)



######## IMPACTFACE LABEL ##############
dataImpactface = df.drop(["grip", "address", "backcock", "top", "downBody", "downCock", "impactPass", "followSwing", "followTop"], axis=1)
dataImpactfaceTrainset, dataImpactfaceTestset = split_to_train_test(dataImpactface, 'impactFace', 0.9)
dataImpactfaceTrainset = dataImpactfaceTrainset.to_numpy()
dataImpactfaceTestset = dataImpactfaceTestset.to_numpy()
dataImpactfaceTrainD, dataImpactfaceTrainL = dataImpactfaceTrainset[:, :24], dataImpactfaceTrainset[:, 24:]
dataImpactfaceTestD, dataImpactfaceTestL = dataImpactfaceTestset[:, :24], dataImpactfaceTestset[:, 24:]
scaler = StandardScaler()
dataImpactfaceTrainD_scaled = scaler.fit_transform(dataImpactfaceTrainD)
train_with_xgboost(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_lightGBM(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)



######## IMPACTPASS LABEL ##############
dataImpactpass = df.drop(["grip", "address", "backcock", "top", "downBody", "downCock", "impactFace", "followSwing", "followTop"], axis=1)
dataImpactpassTrainset, dataImpactpassTestset = split_to_train_test(dataImpactpass, 'impactPass', 0.9)
dataImpactpassTrainset = dataImpactpassTrainset.to_numpy()
dataImpactpassTestset = dataImpactpassTestset.to_numpy()
dataImpactpassTrainD, dataImpactpassTrainL = dataImpactpassTrainset[:, :24], dataImpactpassTrainset[:, 24:]
dataImpactpassTestD, dataImpactpassTestL = dataImpactpassTestset[:, :24], dataImpactpassTestset[:, 24:]
scaler = StandardScaler()
dataImpactpassTrainD_scaled = scaler.fit_transform(dataImpactpassTrainD)
train_with_xgboost(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_lightGBM(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)



######## FOLLOWSWING LABEL ##############
dataFollowswing = df.drop(["grip", "address", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass",  "followTop"], axis=1)
dataFollowswingTrainset, dataFollowswingTestset = split_to_train_test(dataFollowswing, 'followSwing', 0.9)
dataFollowswingTrainset = dataFollowswingTrainset.to_numpy()
dataFollowswingTestset = dataFollowswingTestset.to_numpy()
dataFollowswingTrainD, dataFollowswingTrainL = dataFollowswingTrainset[:, :24], dataFollowswingTrainset[:, 24:]
dataFollowswingTestD, dataFollowswingTestL = dataFollowswingTestset[:, :24], dataFollowswingTestset[:, 24:]
scaler = StandardScaler()
dataFollowswingTrainD_scaled = scaler.fit_transform(dataFollowswingTrainD)
train_with_xgboost(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_lightGBM(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)



######## FOLLOWTOP LABEL ##############
dataFollowtop = df.drop(["grip", "address", "backcock", "top", "downBody", "downCock", "impactFace", "impactPass", "followSwing"], axis=1)
dataFollowtopTrainset, dataFollowtopTestset = split_to_train_test(dataFollowtop, 'followTop', 0.9)
dataFollowtopTrainset = dataFollowtopTrainset.to_numpy()
dataFollowtopTestset = dataFollowtopTestset.to_numpy()
dataFollowtopTrainD, dataFollowtopTrainL = dataFollowtopTrainset[:, :24], dataFollowtopTrainset[:, 24:]
dataFollowtopTestD, dataFollowtopTestL = dataFollowtopTestset[:, :24], dataFollowtopTestset[:, 24:]
scaler = StandardScaler()
dataFollowtopTrainD_scaled = scaler.fit_transform(dataFollowtopTrainD)
train_with_xgboost(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_lightGBM(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)











