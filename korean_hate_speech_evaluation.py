from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------- print stat -------------------------------- #
def print_information(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    labels = set(predictions)

    for label in labels:
        print()
        print("Stat of the {} Class".format(label))
        # print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label=label)))
        # print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label=label)))
        # print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label)))
        print("Recall {}".format(recall_score(real_values, predictions, labels=labels, average='micro')))
        print("Precision {}".format(precision_score(real_values, predictions, labels=labels, average='micro')))
        print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, average='micro')))

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))


def print_information_multi_class(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    labels = set(predictions)

    # for label in labels:
    #     print()
    #     print("Stat of the {} Class".format(label))
    #     print("Recall {}".format(recall_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))
    #     print("Precision {}".format(precision_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))
    #     print("F1 Score {}".format(f1_score(real_values, predictions, labels=labels, pos_label=label, average='weighted')))

    print()
    print("Weighted Recall {}".format(recall_score(real_values, predictions, average='weighted')))
    print("Weighted Precision {}".format(precision_score(real_values, predictions, average='weighted')))
    print("Weighter F1 Score {}".format(f1_score(real_values, predictions, average='weighted')))

    print("Macro F1 Score {}".format(f1_score(real_values, predictions, average='macro')))

# ---------------------------------------------------------------------------- #

le = LabelEncoder()

def encode(data):
    return le.fit_transform(data)


def decode(data):
    return le.inverse_transform(data)

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')