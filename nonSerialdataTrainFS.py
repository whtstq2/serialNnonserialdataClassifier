
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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


data_path = 'C:/Users/whtstq2/Desktop/smartGolf/notserial/nonserialdata_other.xlsx'

df = pd.read_excel(data_path)

'''
df = df.drop(['Type', 'Sex', 'Height', 'Weight', 'Hand', 'Club', 'GameClub', 'Ball', 'saveTime', 'SpinLoft', 'TotalTime'], axis=1)
'''


######## GRIP LABEL ##############
# one label data
dataGrip = df.drop(["Address", "Back", "Top", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)

# data ratio split
dataGripTrainset, dataGripTestset = split_to_train_test(dataGrip, 'Grip', 0.8)
print(np.shape(dataGripTrainset), np.shape(dataGripTestset))

# convert dataframe to numpy
dataGripTrainset = dataGripTrainset.to_numpy()
dataGripTestset = dataGripTestset.to_numpy()

# train, testset label slicing
dataGripTrainD, dataGripTrainL = dataGripTrainset[:, :20], dataGripTrainset[:, 20:]
print(np.shape(dataGripTrainD), np.shape(dataGripTrainL))
dataGripTestD, dataGripTestL = dataGripTestset[:, :20], dataGripTestset[:, 20:]
print(np.shape(dataGripTestD), np.shape(dataGripTestL))

print(np.any(np.isnan(dataGripTrainD)))

# train data standardization
scaler = StandardScaler()
dataGripTrainD_scaled = scaler.fit_transform(dataGripTrainD)

# train
train_with_xgboost(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_lightGBM(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_svm(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_mlp(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)
train_with_rf(dataGripTrainD, dataGripTrainL, dataGripTestD, dataGripTestL)

######## ADDRESS LABEL ##############
dataAddress = df.drop(["Grip", "Back", "Top", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataAddressTrainset, dataAddressTestset = split_to_train_test(dataAddress, 'Address', 0.8)
dataAddressTrainset = dataAddressTrainset.to_numpy()
dataAddressTestset = dataAddressTestset.to_numpy()
dataAddressTrainD, dataAddressTrainL = dataAddressTrainset[:, :20], dataAddressTrainset[:, 20:]
dataAddressTestD, dataAddressTestL = dataAddressTestset[:, :20], dataAddressTestset[:, 20:]
scaler = StandardScaler()
dataAddressTrainD_scaled = scaler.fit_transform(dataAddressTrainD)
train_with_xgboost(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_lightGBM(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_svm(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_mlp(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)
train_with_rf(dataAddressTrainD, dataAddressTrainL, dataAddressTestD, dataAddressTestL)

######## BACKCOCK LABEL ##############
dataBackcock = df.drop(["Grip", "Address", "Top", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataBackcockTrainset, dataBackcockTestset = split_to_train_test(dataBackcock, 'Back', 0.8)
dataBackcockTrainset = dataBackcockTrainset.to_numpy()
dataBackcockTestset = dataBackcockTestset.to_numpy()
dataBackcockTrainD, dataBackcockTrainL = dataBackcockTrainset[:, :20], dataBackcockTrainset[:, 20:]
dataBackcockTestD, dataBackcockTestL = dataBackcockTestset[:, :20], dataBackcockTestset[:, 20:]
scaler = StandardScaler()
dataBackcockTrainD_scaled = scaler.fit_transform(dataBackcockTrainD)
train_with_xgboost(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_lightGBM(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_svm(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_mlp(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)
train_with_rf(dataBackcockTrainD, dataBackcockTrainL, dataBackcockTestD, dataBackcockTestL)


######## TOP LABEL ##############
dataTop = df.drop(["Grip", "Address", "Back", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataTopTrainset, dataTopTestset = split_to_train_test(dataTop, 'Top', 0.8)
dataTopTrainset = dataTopTrainset.to_numpy()
dataTopTestset = dataTopTestset.to_numpy()
dataTopTrainD, dataTopTrainL = dataTopTrainset[:, :20], dataTopTrainset[:, 20:]
dataTopTestD, dataTopTestL = dataTopTestset[:, :20], dataTopTestset[:, 20:]
scaler = StandardScaler()
dataTopTrainD_scaled = scaler.fit_transform(dataTopTrainD)
train_with_xgboost(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_lightGBM(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_svm(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_mlp(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)
train_with_rf(dataTopTrainD, dataTopTrainL, dataTopTestD, dataTopTestL)


######## DOWNBODY LABEL ##############
dataDownbody = df.drop(["Grip", "Address", "Back", "Top", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataDownbodyTrainset, dataDownbodyTestset = split_to_train_test(dataDownbody, 'DownBody', 0.8)
dataDownbodyTrainset = dataDownbodyTrainset.to_numpy()
dataDownbodyTestset = dataDownbodyTestset.to_numpy()
dataDownbodyTrainD, dataDownbodyTrainL = dataDownbodyTrainset[:, :20], dataDownbodyTrainset[:, 20:]
dataDownbodyTestD, dataDownbodyTestL = dataDownbodyTestset[:, :20], dataDownbodyTestset[:, 20:]
scaler = StandardScaler()
dataDownbodyTrainD_scaled = scaler.fit_transform(dataDownbodyTrainD)
train_with_xgboost(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_lightGBM(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_svm(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_mlp(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)
train_with_rf(dataDownbodyTrainD, dataDownbodyTrainL, dataDownbodyTestD, dataDownbodyTestL)


######## DOWNCOCK LABEL ##############
dataDowncock = df.drop(["Grip", "Address", "Back", "Top", "DownBody","ImpactFace", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataDowncockTrainset, dataDowncockTestset = split_to_train_test(dataDowncock, 'DownCock', 0.8)
dataDowncockTrainset = dataDowncockTrainset.to_numpy()
dataDowncockTestset = dataDowncockTestset.to_numpy()
dataDowncockTrainD, dataDowncockTrainL = dataDowncockTrainset[:, :20], dataDowncockTrainset[:, 20:]
dataDowncockTestD, dataDowncockTestL = dataDowncockTestset[:, :20], dataDowncockTestset[:, 20:]
scaler = StandardScaler()
dataDowncockTrainD_scaled = scaler.fit_transform(dataDowncockTrainD)
train_with_xgboost(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_lightGBM(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_svm(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_mlp(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)
train_with_rf(dataDowncockTrainD, dataDowncockTrainL, dataDowncockTestD, dataDowncockTestL)


######## IMPACTFACE LABEL ##############
dataImpactface = df.drop(["Grip", "Address", "Back", "Top", "DownBody", "DownCock", "ImpactPass", "FollowSwing", "FollowTop"], axis=1)
dataImpactfaceTrainset, dataImpactfaceTestset = split_to_train_test(dataImpactface, 'ImpactFace', 0.8)
dataImpactfaceTrainset = dataImpactfaceTrainset.to_numpy()
dataImpactfaceTestset = dataImpactfaceTestset.to_numpy()
dataImpactfaceTrainD, dataImpactfaceTrainL = dataImpactfaceTrainset[:, :20], dataImpactfaceTrainset[:, 20:]
dataImpactfaceTestD, dataImpactfaceTestL = dataImpactfaceTestset[:, :20], dataImpactfaceTestset[:, 20:]
scaler = StandardScaler()
dataImpactfaceTrainD_scaled = scaler.fit_transform(dataImpactfaceTrainD)
train_with_xgboost(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_lightGBM(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_svm(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_mlp(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)
train_with_rf(dataImpactfaceTrainD, dataImpactfaceTrainL, dataImpactfaceTestD, dataImpactfaceTestL)


######## IMPACTPASS LABEL ##############
dataImpactpass = df.drop(["Grip", "Address", "Back", "Top", "DownBody", "DownCock", "ImpactFace", "FollowSwing", "FollowTop"], axis=1)
dataImpactpassTrainset, dataImpactpassTestset = split_to_train_test(dataImpactpass, 'ImpactPass', 0.8)
dataImpactpassTrainset = dataImpactpassTrainset.to_numpy()
dataImpactpassTestset = dataImpactpassTestset.to_numpy()
dataImpactpassTrainD, dataImpactpassTrainL = dataImpactpassTrainset[:, :20], dataImpactpassTrainset[:, 20:]
dataImpactpassTestD, dataImpactpassTestL = dataImpactpassTestset[:, :20], dataImpactpassTestset[:, 20:]
scaler = StandardScaler()
dataImpactpassTrainD_scaled = scaler.fit_transform(dataImpactpassTrainD)
train_with_xgboost(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_lightGBM(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_svm(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_mlp(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)
train_with_rf(dataImpactpassTrainD, dataImpactpassTrainL, dataImpactpassTestD, dataImpactpassTestL)


######## FOLLOWSWING LABEL ##############
dataFollowswing = df.drop(["Grip", "Address", "Back", "Top", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowTop"], axis=1)
dataFollowswingTrainset, dataFollowswingTestset = split_to_train_test(dataFollowswing, 'FollowSwing', 0.8)
dataFollowswingTrainset = dataFollowswingTrainset.to_numpy()
dataFollowswingTestset = dataFollowswingTestset.to_numpy()
dataFollowswingTrainD, dataFollowswingTrainL = dataFollowswingTrainset[:, :20], dataFollowswingTrainset[:, 20:]
dataFollowswingTestD, dataFollowswingTestL = dataFollowswingTestset[:, :20], dataFollowswingTestset[:, 20:]
scaler = StandardScaler()
dataFollowswingTrainD_scaled = scaler.fit_transform(dataFollowswingTrainD)
train_with_xgboost(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_lightGBM(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_svm(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_mlp(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)
train_with_rf(dataFollowswingTrainD, dataFollowswingTrainL, dataFollowswingTestD, dataFollowswingTestL)


######## FOLLOWTOP LABEL ##############
dataFollowtop = df.drop(["Grip", "Address", "Back", "Top", "DownBody", "DownCock", "ImpactFace", "ImpactPass", "FollowSwing"], axis=1)
dataFollowtopTrainset, dataFollowtopTestset = split_to_train_test(dataFollowtop, 'FollowTop', 0.8)
dataFollowtopTrainset = dataFollowtopTrainset.to_numpy()
dataFollowtopTestset = dataFollowtopTestset.to_numpy()
dataFollowtopTrainD, dataFollowtopTrainL = dataFollowtopTrainset[:, :20], dataFollowtopTrainset[:, 20:]
dataFollowtopTestD, dataFollowtopTestL = dataFollowtopTestset[:, :20], dataFollowtopTestset[:, 20:]
scaler = StandardScaler()
dataFollowtopTrainD_scaled = scaler.fit_transform(dataFollowtopTrainD)
train_with_xgboost(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_lightGBM(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_svm(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_mlp(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)
train_with_rf(dataFollowtopTrainD, dataFollowtopTrainL, dataFollowtopTestD, dataFollowtopTestL)










