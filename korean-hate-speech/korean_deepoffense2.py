import os
import shutil
import time
import csv
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
import sys        
 
sys.path.append('/root/team26/DeepOffense')
from classification_model import ClassificationModel
from deepoffense.classification import MultiLabelClassificationModel
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from examples.common.download import download_from_google_drive
from korean_deepoffense_config import LANGUAGE_FINETUNE, TEMP_DIRECTORY, SUBMISSION_FOLDER, \
    MODEL_TYPE, MODEL_NAME, language_modeling_args, args, SEED, RESULT_FILE, GOOGLE_DRIVE, DRIVE_FILE_ID
from common.evaluation import macro_f1, weighted_f1
from common.label_converter import decode, encode
from common.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)
if not os.path.exists(os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(TEMP_DIRECTORY, SUBMISSION_FOLDER))

train = pd.read_csv('/root/team26/DeepOffense/examples/korean/data/train.tsv', sep="\t")
train = train.rename(columns={'comments': 'text', 'hate': 'labels'})
train = train[['text', 'labels']]
train['labels'] = train['labels'].replace(['hate','offensive','none'],[0,1,2])


train2 = pd.read_csv('/root/team26/DeepOffense/examples/korean/data/labeled_data.csv', sep=",")
train2 = train2.rename(columns={'tweet': 'text', 'class': 'labels'})
train2 = train2[['text', 'labels']]
for i in range(len(train2['text'])):
    temp= train2['text'][i].strip('"')
    temp = " ".join(filter(lambda x:x[0]!='@', temp.split()))
    temp = " ".join(filter(lambda x:x[0]!='&', temp.split()))
    temp = " ".join(filter(lambda x:x[0:4]!='http', temp.split()))
    temp = " ".join(filter(lambda x:x[0:2]!='RT', temp.split()))
    train2.loc[i, 'text'] = temp

# train2.loc[(train2['labels'] == 0), 'labels'] = 'hate'
# train2.loc[(train2['labels'] == 1), 'labels'] = 'offensive'
# train2.loc[(train2['labels'] == 2), 'labels'] = 'none'
# train.loc[(train['labels'] == 'offensive'), 'labels'] = 'hate'

test = pd.read_csv('/root/team26/DeepOffense/examples/korean/data/test.tsv', sep="\t")
test = test.rename(columns={'comments': 'text', 'hate': 'labels'})
test = test[['text', 'labels']]
test['labels'] = test['labels'].replace(['hate','offensive','none'],[0,1,2])

# test.loc[(test['labels'] == 'offensive'), 'labels'] = 'hate'

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

if LANGUAGE_FINETUNE:
    train_list = train['text'].tolist()
    test_list = test['text'].tolist()
    complete_list = train_list + test_list
    lm_train = complete_list[0: int(len(complete_list)*0.8)]
    lm_test = complete_list[-int(len(complete_list)*0.2):]

    with open(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), 'w') as f:
        for item in lm_train:
            f.write("%s\n" % item)

    with open(os.path.join(TEMP_DIRECTORY, "lm_test.txt"), 'w') as f:
        for item in lm_test:
            f.write("%s\n" % item)

    model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=language_modeling_args)
    model.train_model(os.path.join(TEMP_DIRECTORY, "lm_train.txt"), eval_file=os.path.join(TEMP_DIRECTORY, "lm_test.txt"))
    MODEL_NAME = language_modeling_args["best_model_dir"]


# Train the model
print("Started Training")

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test), args["n_fold"]))


eng_test_preds = np.zeros((len(test), 1))
# ENGLISH PRETRAIN
if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
    shutil.rmtree(args['output_dir'])
print("ENGLISH PRETRAIN")

# model = ClassificationModel(MODEL_TYPE, MODEL_NAME, num_labels=3, args=args,
model = ClassificationModel(MODEL_TYPE, MODEL_NAME, num_labels=3, args=args,
                            use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
train_df, eval_df = train_test_split(train2, test_size=0.1, random_state=SEED * 42)
# model.train_model(train_df,multi_label=True, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3, args=args,
                            use_cuda=torch.cuda.is_available())

eng_predictions, eng_raw_outputs = model.predict(test_sentences)
# test_preds[:, 0] = predictions

print("COMPLETED PRETRAIN")


if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))
        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, num_labels=3, args=args,
                                    use_cuda=torch.cuda.is_available())  # You can set class weights by using the optional weight argument
        # tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        # model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased')
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)

        # model.train_model(train_df,multi_label=True, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3, args=args,
                                    use_cuda=torch.cuda.is_available())

        predictions, raw_outputs = model.predict(test_sentences)
        test_preds[:, i] = predictions
        print("Completed Fold {}".format(i))
    # select majority class of each instance (row)
    final_predictions = []
    for row in test_preds:
        row = row.tolist()
        final_predictions.append(int(max(set(row), key=row.count)))
    test['predictions'] = final_predictions
else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, num_labels=3, args=args,
                                    use_cuda=torch.cuda.is_available())
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3, args=args,
                                    use_cuda=torch.cuda.is_available())
    predictions, raw_outputs = model.predict(test_sentences)
    test['predictions'] = predictions

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

time.sleep(5)

test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')
print_information(test, "predictions", "labels")


