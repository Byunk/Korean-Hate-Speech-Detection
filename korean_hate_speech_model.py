#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import time
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split
import sys

sys.path.append('./DeepOffense')
from deepoffense.classification import ClassificationModel
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel
from korean_hate_speech_config import (
    LANGUAGE_FINETUNE, 
    PATH_RESULT,
    PATH_DATA,
    PATH_PRETRAIN,
    SUBMISSION_FOLDER,
    language_modeling_args,
    args,
    SEED, 
    RESULT_FILE,
    MODEL_TYPE
)
from korean_hate_speech_evaluation import (
    macro_f1, weighted_f1, decode, encode, print_information
)

from transformers import XLMRobertaTokenizer

if not os.path.exists(PATH_RESULT): os.makedirs(PATH_RESULT)
if not os.path.exists(os.path.join(PATH_RESULT, SUBMISSION_FOLDER)): os.makedirs(
    os.path.join(PATH_RESULT, SUBMISSION_FOLDER))

class KODEMODEL:

    model_size_class = {
        'large': 'xlm-roberta-large',
        'base': 'xlm-roberta-base'
    }

    encoding_class = [
        'aboslute', 'relative'
    ]

    tokenizer_class = [
        'khaiii'
    ]

    pretrain_method_class = [
        'eng',
        'sandwich'
    ]

    def __init__(
            self,
            model_size = 'large',
            positional_encoding = 'absolute',
            tokenizer = None,
            pretrain_method = None,
    ):
        self.model_name = self.model_size_class[model_size];
        self.pretrain_method = pretrain_method;
        self.tokenizer = tokenizer
        if self.pretrain_method and self.pretrain_method not in self.pretrain_method_class:
            print("INVALID Pretrain method")
            self.pretrain_method = None
        elif self.pretrain_method:
            self.pretrain_path = os.path.join(PATH_PRETRAIN, pretrain_method)
        else:
            self.pretrain_path = os.path.join(PATH_PRETRAIN, 'no_pretrain')

        if positional_encoding == 'relative':
            args['config'] = {"position_embedding_type": "relative_key"}
        

    def train(self):
        # Preprocessing
        train = pd.read_csv(PATH_DATA + '/augmented.csv', sep=",")
        train = train.rename(columns={'comments': 'text', 'hate': 'labels'})
        train = train[['text', 'labels']]
        train['labels'] = train['labels'].replace(['hate','offensive','none'],[0,1,2])

        train2 = pd.read_csv(PATH_DATA + '/labeled_data.csv', sep=",")
        train2 = train2.rename(columns={'tweet': 'text', 'class': 'labels'})
        train2 = train2[['text', 'labels']]
        for i in range(len(train2['text'])):
            temp= train2['text'][i].strip('"')
            temp = " ".join(filter(lambda x:x[0]!='@', temp.split()))
            temp = " ".join(filter(lambda x:x[0]!='&', temp.split()))
            temp = " ".join(filter(lambda x:x[0:4]!='http', temp.split()))
            temp = " ".join(filter(lambda x:x[0:2]!='RT', temp.split()))
            train2.loc[i, 'text'] = temp

        test = pd.read_csv(PATH_DATA + '/test.tsv', sep="\t")
        test = test.rename(columns={'comments': 'text', 'hate': 'labels'})
        test = test[['text', 'labels']]
        test['labels'] = test['labels'].replace(['hate','offensive','none'],[0,1,2])

        if LANGUAGE_FINETUNE:
            print("Language Finetune")
            train_list = train['text'].tolist()
            test_list = test['text'].tolist()
            complete_list = train_list + test_list
            lm_train = complete_list[0: int(len(complete_list)*0.8)]
            lm_test = complete_list[-int(len(complete_list)*0.2):]

            with open(os.path.join(PATH_RESULT, "lm_train.txt"), 'w') as f:
                for item in lm_train:
                    f.write("%s\n" % item)

            with open(os.path.join(PATH_RESULT, "lm_test.txt"), 'w') as f:
                for item in lm_test:
                    f.write("%s\n" % item)

            model = LanguageModelingModel(MODEL_TYPE, self.model_name, args=language_modeling_args)
            model.train_model(os.path.join(PATH_RESULT, "lm_train.txt"), eval_file=os.path.join(PATH_RESULT, "lm_test.txt"))
            self.model_name = language_modeling_args["best_model_dir"]

        print("Start Training...")

        train['labels'] = encode(train["labels"])
        test['labels'] = encode(test["labels"])
        test_sentences = test['text'].tolist()
        test_preds = np.zeros((len(test), args["n_fold"]))

        # ENGLISH PRETRAIN
        if self.pretrain_method and not os.path.exists(self.pretrain_path):
            # if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            #     shutil.rmtree(args['output_dir'])
            print("Start ENGLISH PRETRAIN...")

            args['output_dir'] = os.path.join(PATH_PRETRAIN, "eng")
            model = ClassificationModel(MODEL_TYPE, self.model_name, num_labels=3, args=args,
                                        use_cuda=torch.cuda.is_available())
            if self.tokenizer:
                model.tokenizer = XLMRobertaTokenizer.from_pretrained("/root/team26/DeepOffense/examples/korean/my_xlmr", do_lower_case=self.args.do_lower_case, **kwargs)
            train_df, eval_df = train_test_split(train2, test_size=0.1, random_state=SEED * 42)
            model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
            # model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3, args=args,
            #                             use_cuda=torch.cuda.is_available())
            self.model_name = os.path.join(PATH_PRETRAIN, "eng")
            # os.rename(args['output_dir'], MODEL_NAME)
            
            if self.pretrain_method == 'sandwich':
                print("Start Sandwich PRETRAIN...")

                args['output_dir'] = self.pretrain_path
                model = ClassificationModel(self.model_name, MODEL_NAME, num_labels=3, args=args,use_cuda=torch.cuda.is_available()) 
                train_df, eval_df = train_test_split(train2, test_size=0.1, random_state=SEED * 42)
                model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
        elif self.pretrain_method:
            self.model_name = self.pretrain_path

        args['output_dir'] = PATH_RESULT

        if args["evaluate_during_training"]:
            for i in range(args["n_fold"]):
                if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
                    shutil.rmtree(args['output_dir'])
                print("Started Fold {}".format(i))
                model = ClassificationModel(MODEL_TYPE, self.model_name, num_labels=3, args=args,
                                            use_cuda=torch.cuda.is_available()) 
                train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
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
            model = ClassificationModel(MODEL_TYPE, self.model_name, num_labels=3, args=args,
                                            use_cuda=torch.cuda.is_available())
            model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
            model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], num_labels=3, args=args,
                                            use_cuda=torch.cuda.is_available())
            predictions, raw_outputs = model.predict(test_sentences)
            test['predictions'] = predictions

        test['predictions'] = decode(test['predictions'])
        test['labels'] = decode(test['labels'])

        time.sleep(5)

        test.to_csv(os.path.join(PATH_RESULT, RESULT_FILE),  header=True, sep='\t', index=False, encoding='utf-8')
        print_information(test, "predictions", "labels")
