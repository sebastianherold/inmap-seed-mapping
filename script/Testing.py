import pandas as pd
from Metrics import Metric
from sklearn.model_selection import StratifiedShuffleSplit
import random

def evaluate_classifier(
    dataFrame,
    classifier,
    feature_representation,
    fold_quantity,
    type,
    test_size=0.9,
    number_of_files_for_training=5,
    df_training=None
):
    if type == "standard":
        return evaluate_classifier_standard(
            dataFrame, classifier, feature_representation, fold_quantity, test_size)
    elif type == "custom":
        return evaluate_classifier_custom(
            dataFrame, classifier, feature_representation, fold_quantity, number_of_files_for_training)
    elif type == "split":
        return evaluate_classifier_split(dataFrame, classifier, feature_representation, df_training)


def evaluate_classifier_standard(
    dataFrame, classifier, feature_representation, fold_quantity, test_size
):

    classifierMetrics = []
    
    y = dataFrame.Label
    X = dataFrame.FileContent

    skf = StratifiedShuffleSplit(
        n_splits=fold_quantity, test_size=test_size, random_state=55
    )

    for index_train, index_test in skf.split(X, y):
        x_train_fold, x_test_fold = X[index_train], X[index_test]
        y_train_fold, y_test_fold = y[index_train], y[index_test]
        X_transformer_train = feature_representation.fit_transform(x_train_fold)
        X_transformer_test = feature_representation.transform(x_test_fold)
        c = classifier.createObject()
        c.fit(X_transformer_train, y_train_fold)
        y_pred = c.predict(X_transformer_test)
        classifierMetrics.append(
            Metric(
                y_test_fold,
                y_pred,
                X_transformer_test,
                classifier.getClassifierName(),
                dataFrame,
            )
        )
    return classifierMetrics


def evaluate_classifier_custom(
    dataFrame, classifier, feature_representation, fold_quantity, number_of_files_for_training
):
    """
    Pre: The number of files chosen should not exceed the number of actual existing files in the system,
    this test is done to simulate real scenario where someone wants to take a portion of files (evenly)
    and map to corresponding label
    """

    classifierMetrics = []

    for index in range(fold_quantity):

        x_train, x_test, y_train, y_test = train_fold_quantity_custom(
            dataFrame, number_of_files_for_training, random.randint(0, fold_quantity)
        )
        X_transformer_train = feature_representation.fit_transform(x_train)
        X_transformer_test = feature_representation.transform(x_test)
        c = classifier.createObject()
        c.fit(X_transformer_train, y_train)
        y_pred = c.predict(X_transformer_test)
        classifierMetrics.append(
            Metric(
                y_test,
                y_pred,
                X_transformer_test,
                classifier.getClassifierName(),
                dataFrame,
            )
        )
    return classifierMetrics

def evaluate_classifier_split(dataFrame, classifier, feature_representation, df_training_inmap):
    classifierMetrics = []
    df_training = pd.DataFrame()
    df_training['FileName'] = df_training_inmap['name'].astype(str)
    df_training = pd.merge(df_training, dataFrame, on='FileName', how='left')
    df_test = dataFrame[dataFrame.FileName.isin(df_training.FileName) == False]
    x_train = feature_representation.fit_transform(df_training['FileContent'])
    x_test = feature_representation.transform(df_test['FileContent'])
    c = classifier.createObject()
    c.fit(x_train, df_training['Label'])
    y_pred = c.predict(x_test)
    classifierMetrics.append(Metric(
        df_test['Label'],
        y_pred,
        x_test,
        classifier.getClassifierName(),
        dataFrame))
    return classifierMetrics

def train_fold_quantity_custom(dataFrame, number_of_files_for_training, RNG):
    listOfLabels = dataFrame.Label.unique()
    listOfLabels.sort()
    TrainingFrame = pd.DataFrame(columns=dataFrame.columns)
    for currLabel in listOfLabels:
        currentLabelFrame = dataFrame.loc[dataFrame["Label"] == currLabel]
        selectedFrame = currentLabelFrame.sample(
            number_of_files_for_training, random_state= RNG
        )
        TrainingFrame = TrainingFrame.append(selectedFrame, ignore_index=True)
        indexArray = selectedFrame.index.tolist()
        dataFrame = dataFrame.drop(indexArray)

    dataFrame = dataFrame.reset_index()  # Uppdaterar index, startar från 0

    x_train = TrainingFrame["FileContent"]
    y_train = TrainingFrame["Label"]

    x_test = dataFrame["FileContent"]
    y_test = dataFrame["Label"]
    return x_train, x_test, y_train, y_test
