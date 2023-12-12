# General imports
import os
import pickle
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pynvml import *
import shutil
import advertools as adv
import preprocessor as p
import re
import logging
from textblob.blob import Sentence
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix
import random

# VADER imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# BERT imports
from transformers import BertTokenizer, BertModel
from transformers import logging
logging.set_verbosity_error()



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class Construct_SST2_dataset(object):

    def findWholeWord(self, w):

        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def conjunction_analysis(self, sentence):

        tokenized_sentence = tokenizer.tokenize(sentence)

        if 'but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1: # Check if the sentence contains A-but-B structure

            A_clause = sentence.split('but')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('but')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return A_clause, B_clause

        else:
            return None, None

    def preprocess(self):

        pass

    def rule_labeling(self): # A-but-B rule labeling of SST2 sentences through VADER
        
        print("\n")

        # SST2 sentences
        total_no_of_sentences = 0
        total_no_of_positive_sentiment_sentences = 0
        total_no_of_negative_sentiment_sentences = 0
        ground_truth_sentiments_of_sentences = []
        vader_sentiment_predictions_of_sentences = []
        vader_sentiment_corrects_of_sentences = []
        vader_positive_sentiment_corrects_of_sentences = []
        vader_negative_sentiment_corrects_of_sentences = []
        
        # SST2 sentences containing A-but-B structure
        total_no_of_sentences_containing_A_but_B_structure = 0
        total_no_of_positive_sentiment_A_but_B_structure_sentences = 0
        total_no_of_negative_sentiment_A_but_B_structure_sentences = 0
        ground_truth_sentiments_of_A_but_B_structure_sentences = []
        vader_sentiment_predictions_of_A_but_B_structure_sentences = []
        vader_sentiment_corrects_of_A_but_B_structure_sentences = []
        vader_positive_sentiment_corrects_of_A_but_B_structure_sentences = []
        vader_negative_sentiment_corrects_of_A_but_B_structure_sentences = []

        # VADER scores
        vader_sentiment_scores = {"sentence":[], "clause_A":[], "clause_B":[]}
        
        # VADER distribution for SST2 sentences
        Contrast_A_but_B_rule_valid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        Contrast_A_but_B_rule_invalid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        No_contrast_A_but_B_rule_valid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        No_contrast_A_but_B_rule_invalid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}

        # VADER distribution for SST2 sentences for which it correctly predicts the sentence sentiment
        Correct_contrast_A_but_B_rule_valid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        Correct_contrast_A_but_B_rule_invalid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        Correct_no_contrast_A_but_B_rule_valid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}
        Correct_no_contrast_A_but_B_rule_invalid = {'sentence':[], 'clause_A':[], 'clause_B':[], 'ground_truth_sentiment':[], 'vader_sentence_sentiment':[], 'vader_clause_A_sentiment':[], 'vader_clause_B_sentiment':[]}

        # Accuracy of B clause
        B_clause_corrects = []

        # Accuracy of B clause and sentences
        B_clause_sentences_corrects = []

        count = 0

        # Select a datapoint from the file
        preprocessed_data_path = "datasets/SST2_sentences/preprocessed_dataset/stsa.binary.p"
        preprocessed_data = pickle.load(open(preprocessed_data_path, "rb"))
        revs, pre_trained_word_vectors, random_word_vectors, word_idx_map, vocab = preprocessed_data[0], preprocessed_data[1], preprocessed_data[2], preprocessed_data[3], preprocessed_data[4]

        with tqdm(desc = "SST2 dataset", total = 9613) as pbar:
            for index, rev in enumerate(revs):

                # Extract sentence and sentiment from the datapoint
                sentence = rev["text"]
                ground_truth_sentiment = rev['y'] # -1 => Negative, 1 => Positive and 0 => Neutral
                if ground_truth_sentiment == 0:
                    ground_truth_sentiment = -1
                    total_no_of_negative_sentiment_sentences += 1
                elif ground_truth_sentiment == 1:
                    ground_truth_sentiment = 1
                    total_no_of_positive_sentiment_sentences += 1
                ground_truth_sentiments_of_sentences.append(ground_truth_sentiment)
                
                # Calculate the VADER sentiment score for the sentence
                analyzer = SentimentIntensityAnalyzer() 
                vader_score_sentence = analyzer.polarity_scores(sentence)
                vader_sentiment_scores['sentence'].append(vader_score_sentence)

                # Assign the sentiment class as per the sentiment scores (using default interpretation as per the authors of VADER)
                if vader_score_sentence['compound'] >= 0.05:
                    vader_sentiment_sentence = 1 # Positive sentiment
                    vader_sentiment_predictions_of_sentences.append(vader_sentiment_sentence)
                elif (vader_score_sentence['compound'] > -0.05) and (vader_score_sentence['compound'] < 0.05):
                    vader_sentiment_sentence = 0 # Neutral sentiment
                    vader_sentiment_predictions_of_sentences.append(vader_sentiment_sentence)
                elif vader_score_sentence["compound"] <= -0.05:
                    vader_sentiment_sentence = -1 # Negative sentiment
                    vader_sentiment_predictions_of_sentences.append(vader_sentiment_sentence)
                
                # Check if the sentence has been correctly classified by VADER
                if vader_sentiment_sentence == ground_truth_sentiment:
                    vader_sentiment_corrects_of_sentences.append(1)
                    if ground_truth_sentiment == 1:
                        vader_positive_sentiment_corrects_of_sentences.append(1)
                    elif ground_truth_sentiment == -1:
                        vader_negative_sentiment_corrects_of_sentences.append(1)
                elif vader_sentiment_sentence != ground_truth_sentiment:
                    vader_sentiment_corrects_of_sentences.append(0)
                    if ground_truth_sentiment == 1:
                        vader_positive_sentiment_corrects_of_sentences.append(0)
                    elif ground_truth_sentiment == -1:
                        vader_negative_sentiment_corrects_of_sentences.append(0)

                # Conjunction analysis A-but-B structure                             
                clause_A, clause_B = self.conjunction_analysis(sentence)
                if clause_A == None and clause_B != None:
                    print(sentence)
                elif clause_A != None and clause_B == None:
                    print(sentence)
                elif (clause_A != None) and (clause_B != None):
                    
                    # Append the values to A-but-B structure sentences
                    total_no_of_sentences_containing_A_but_B_structure += 1

                    ground_truth_sentiments_of_A_but_B_structure_sentences.append(ground_truth_sentiment)
                    if ground_truth_sentiment == 1:
                        total_no_of_positive_sentiment_A_but_B_structure_sentences += 1
                    elif ground_truth_sentiment == -1:
                        total_no_of_negative_sentiment_A_but_B_structure_sentences += 1
                    
                    vader_sentiment_predictions_of_A_but_B_structure_sentences.append(vader_sentiment_sentence)
                    if vader_sentiment_sentence == ground_truth_sentiment:
                        vader_sentiment_corrects_of_A_but_B_structure_sentences.append(1)
                        if ground_truth_sentiment == 1:
                            vader_positive_sentiment_corrects_of_A_but_B_structure_sentences.append(1)
                        elif ground_truth_sentiment == -1:
                            vader_negative_sentiment_corrects_of_A_but_B_structure_sentences.append(1)
                    elif vader_sentiment_sentence != ground_truth_sentiment:
                        vader_sentiment_corrects_of_A_but_B_structure_sentences.append(0)
                        if ground_truth_sentiment == 1:
                            vader_positive_sentiment_corrects_of_A_but_B_structure_sentences.append(0)
                        elif ground_truth_sentiment == -1:
                            vader_negative_sentiment_corrects_of_A_but_B_structure_sentences.append(0)

                    # Calculate the VADER sentiment scores for the clauses
                    vader_score_clause_A = analyzer.polarity_scores(clause_A)
                    vader_sentiment_scores['clause_A'].append(vader_score_clause_A)
                    vader_score_clause_B = analyzer.polarity_scores(clause_B)
                    vader_sentiment_scores["clause_B"].append(vader_score_clause_B)
                    
                    # Assign the sentiment class as per the sentiment scores (using default interpretation as per the authors of VADER)
                    if vader_score_clause_A['compound'] >= 0.05:
                        vader_sentiment_clause_A = 1 # Positive sentiment
                    elif (vader_score_clause_A['compound'] > -0.05) and (vader_score_clause_A['compound'] < 0.05):
                        vader_sentiment_clause_A = 0 # Neutral sentiment
                    elif vader_score_clause_A["compound"] <= -0.05:
                        vader_sentiment_clause_A = -1 # Negative sentiment

                    if vader_score_clause_B['compound'] >= 0.05:
                        vader_sentiment_clause_B = 1 # Positive sentiment
                    elif (vader_score_clause_B['compound'] > -0.05) and (vader_score_clause_B['compound'] < 0.05):
                        vader_sentiment_clause_B = 0 # Neutral sentiment
                    elif vader_score_clause_B["compound"] <= -0.05:
                        vader_sentiment_clause_B = -1 # Negative sentiment

                    # Calculate the distributions for different cases of A-but-B rule
                    if (vader_sentiment_clause_A != vader_sentiment_clause_B) and (vader_sentiment_clause_B==ground_truth_sentiment):
                        Contrast_A_but_B_rule_valid['sentence'].append(sentence)
                        Contrast_A_but_B_rule_valid['clause_A'].append(clause_A)
                        Contrast_A_but_B_rule_valid['clause_B'].append(clause_B)
                        Contrast_A_but_B_rule_valid['ground_truth_sentiment'].append(ground_truth_sentiment)
                        Contrast_A_but_B_rule_valid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                        Contrast_A_but_B_rule_valid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                        Contrast_A_but_B_rule_valid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)
                        if (vader_sentiment_sentence == ground_truth_sentiment):
                            Correct_contrast_A_but_B_rule_valid['sentence'].append(sentence)
                            Correct_contrast_A_but_B_rule_valid['clause_A'].append(clause_A)
                            Correct_contrast_A_but_B_rule_valid['clause_B'].append(clause_B)
                            Correct_contrast_A_but_B_rule_valid['ground_truth_sentiment'].append(ground_truth_sentiment)
                            Correct_contrast_A_but_B_rule_valid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                            Correct_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                            Correct_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)

                    elif (vader_sentiment_clause_A != vader_sentiment_clause_B) and (vader_sentiment_clause_B!=ground_truth_sentiment):
                        Contrast_A_but_B_rule_invalid['sentence'].append(sentence)
                        Contrast_A_but_B_rule_invalid['clause_A'].append(clause_A)
                        Contrast_A_but_B_rule_invalid['clause_B'].append(clause_B)
                        Contrast_A_but_B_rule_invalid['ground_truth_sentiment'].append(ground_truth_sentiment)
                        Contrast_A_but_B_rule_invalid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                        Contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                        Contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)
                        if (vader_sentiment_sentence == ground_truth_sentiment):
                            Correct_contrast_A_but_B_rule_invalid['sentence'].append(sentence)
                            Correct_contrast_A_but_B_rule_invalid['clause_A'].append(clause_A)
                            Correct_contrast_A_but_B_rule_invalid['clause_B'].append(clause_B)
                            Correct_contrast_A_but_B_rule_invalid['ground_truth_sentiment'].append(ground_truth_sentiment)
                            Correct_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                            Correct_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                            Correct_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)

                    elif (vader_sentiment_clause_A == vader_sentiment_clause_B) and (vader_sentiment_clause_B==ground_truth_sentiment):
                        No_contrast_A_but_B_rule_valid['sentence'].append(sentence)
                        No_contrast_A_but_B_rule_valid['clause_A'].append(clause_A)
                        No_contrast_A_but_B_rule_valid['clause_B'].append(clause_B)
                        No_contrast_A_but_B_rule_valid['ground_truth_sentiment'].append(ground_truth_sentiment)
                        No_contrast_A_but_B_rule_valid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                        No_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                        No_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)
                        if (vader_sentiment_sentence == ground_truth_sentiment):
                            Correct_no_contrast_A_but_B_rule_valid['sentence'].append(sentence)
                            Correct_no_contrast_A_but_B_rule_valid['clause_A'].append(clause_A)
                            Correct_no_contrast_A_but_B_rule_valid['clause_B'].append(clause_B)
                            Correct_no_contrast_A_but_B_rule_valid['ground_truth_sentiment'].append(ground_truth_sentiment)
                            Correct_no_contrast_A_but_B_rule_valid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                            Correct_no_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                            Correct_no_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)

                    elif (vader_sentiment_clause_A == vader_sentiment_clause_B) and (vader_sentiment_clause_B!=ground_truth_sentiment):
                        No_contrast_A_but_B_rule_invalid['sentence'].append(sentence)
                        No_contrast_A_but_B_rule_invalid['clause_A'].append(clause_A)
                        No_contrast_A_but_B_rule_invalid['clause_B'].append(clause_B)
                        No_contrast_A_but_B_rule_invalid['ground_truth_sentiment'].append(ground_truth_sentiment)
                        No_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                        No_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                        No_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)
                        if (vader_sentiment_sentence == ground_truth_sentiment):
                            Correct_no_contrast_A_but_B_rule_invalid['sentence'].append(sentence)
                            Correct_no_contrast_A_but_B_rule_invalid['clause_A'].append(clause_A)
                            Correct_no_contrast_A_but_B_rule_invalid['clause_B'].append(clause_B)
                            Correct_no_contrast_A_but_B_rule_invalid['ground_truth_sentiment'].append(ground_truth_sentiment)
                            Correct_no_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'].append(vader_sentiment_sentence)
                            Correct_no_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'].append(vader_sentiment_clause_A)
                            Correct_no_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'].append(vader_sentiment_clause_B)
                            

                pbar.update()
        
        # Contrast and A-but-B rule is valid
        indices = [index for index, element in enumerate(Contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == 1]
        Contrast_A_but_B_rule_valid_positive_sentiment = [Contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == -1]
        Contrast_A_but_B_rule_valid_negative_sentiment = [Contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        
        # Contrast but A-but-B rule is invalid
        indices = [index for index, element in enumerate(Contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == 1]
        Contrast_A_but_B_rule_invalid_positive_sentiment = [Contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == -1]
        Contrast_A_but_B_rule_invalid_negative_sentiment = [Contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]

        # No contrast but A-but-B rule is valid
        indices = [index for index, element in enumerate(No_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == 1]
        No_contrast_A_but_B_rule_valid_positive_sentiment = [No_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(No_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == -1]
        No_contrast_A_but_B_rule_valid_negative_sentiment = [No_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]

        # No contrast and A-but-B rule is invalid
        indices = [index for index, element in enumerate(No_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == 1]
        No_contrast_A_but_B_rule_invalid_positive_sentiment = [No_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(No_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == -1]
        No_contrast_A_but_B_rule_invalid_negative_sentiment = [No_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]

        # Corrects contrast and A-but-B rule is valid
        indices = [index for index, element in enumerate(Correct_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == 1]
        Correct_contrast_A_but_B_rule_valid_positive_sentiment = [Correct_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Correct_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == -1]
        Correct_contrast_A_but_B_rule_valid_negative_sentiment = [Correct_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        
        # Corrects contrast but A-but-B rule is invalid
        indices = [index for index, element in enumerate(Correct_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == 1]
        Correct_contrast_A_but_B_rule_invalid_positive_sentiment = [Correct_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Correct_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == -1]
        Correct_contrast_A_but_B_rule_invalid_negative_sentiment = [Correct_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]

        # Corrects no contrast but A-but-B rule is valid
        indices = [index for index, element in enumerate(Correct_no_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == 1]
        Correct_no_contrast_A_but_B_rule_valid_positive_sentiment = [Correct_no_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Correct_no_contrast_A_but_B_rule_valid['ground_truth_sentiment']) if element == -1]
        Correct_no_contrast_A_but_B_rule_valid_negative_sentiment = [Correct_no_contrast_A_but_B_rule_valid['sentence'][index] for index in indices]

        # No contrast and A-but-B rule is invalid
        indices = [index for index, element in enumerate(Correct_no_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == 1]
        Correct_no_contrast_A_but_B_rule_invalid_positive_sentiment = [Correct_no_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]
        indices = [index for index, element in enumerate(Correct_no_contrast_A_but_B_rule_invalid['ground_truth_sentiment']) if element == -1]
        Correct_no_contrast_A_but_B_rule_invalid_negative_sentiment = [Correct_no_contrast_A_but_B_rule_invalid['sentence'][index] for index in indices]

        print("\n")

        print("Total no of sentences in SST2 dataset: ", total_no_of_sentences)
        print("No of sentences with positive sentiment: ", total_no_of_positive_sentiment_sentences)
        print("No of sentences with negative sentiment: ", total_no_of_negative_sentiment_sentences)
        print("No of sentences classified correctly by VADER: ", vader_sentiment_corrects_of_sentences.count(1))
        print("Acc. of VADER on sentences: ", sum(vader_sentiment_corrects_of_sentences)/len(vader_sentiment_corrects_of_sentences))
        print("No of positive sentiment sentences classified correctly by VADER: ", vader_positive_sentiment_corrects_of_sentences.count(1))
        print("Acc. of VADER on positive sentiment sentences: ", sum(vader_positive_sentiment_corrects_of_sentences)/len(vader_positive_sentiment_corrects_of_sentences))
        print("No of negative sentiment sentences classified correctly by VADER: ", vader_negative_sentiment_corrects_of_sentences.count(1))
        print("Acc: of VADER on negative sentiment sentences: ", sum(vader_negative_sentiment_corrects_of_sentences)/len(vader_negative_sentiment_corrects_of_sentences))

        print("\n")

        print("Total no of sentences containing A-but-B structure in SST2 dataset: ", total_no_of_sentences_containing_A_but_B_structure)
        print("No of sentences with positive sentiment containing A-but-B structure: ", total_no_of_positive_sentiment_A_but_B_structure_sentences)
        print("No of sentences with negative sentiment containing A-but-B structure: ", total_no_of_negative_sentiment_A_but_B_structure_sentences)
        print("No of A-but-B structure sentences classified correctly by VADER: ", vader_sentiment_corrects_of_A_but_B_structure_sentences.count(1))
        print("Acc. of VADER on A-but-B structure sentences: ", sum(vader_sentiment_corrects_of_A_but_B_structure_sentences)/len(vader_sentiment_corrects_of_A_but_B_structure_sentences))
        print("No of positive sentiment sentences containing A-but-B structure classified correctly by VADER: ", vader_positive_sentiment_corrects_of_A_but_B_structure_sentences.count(1))
        print("Acc. of VADER on positive sentiment sentences containing A-but-B structure: ", sum(vader_positive_sentiment_corrects_of_A_but_B_structure_sentences)/len(vader_positive_sentiment_corrects_of_A_but_B_structure_sentences))
        print("No of negative sentiment sentences containing A-but-B structure classified correctly by VADER: ", vader_negative_sentiment_corrects_of_A_but_B_structure_sentences.count(1))
        print("Acc: of VADER on negative sentiment sentences containing A-but-B structure: ", sum(vader_negative_sentiment_corrects_of_A_but_B_structure_sentences)/len(vader_negative_sentiment_corrects_of_A_but_B_structure_sentences))

        print("\n")

        print("Out of all sentences containing A-but-B structure ({}), VADER determines that: -\n".format(total_no_of_sentences_containing_A_but_B_structure))
        print("No of sentences with contrast and contain A-but-B rule: ", len(Contrast_A_but_B_rule_valid['sentence']))
        print("No of positive sentiment sentences with contrast and contain A-but-B rule: ", len(Contrast_A_but_B_rule_valid_positive_sentiment))
        print("No of negative sentiment sentences with contrast and contain A-but-B rule: ", len(Contrast_A_but_B_rule_valid_negative_sentiment))
        print("No of sentences with contrast but do not contain A-but-B rule: ", len(Contrast_A_but_B_rule_invalid['sentence']))
        print("No of positive sentiment sentences with contrast but do not contain A-but-B rule: ", len(Contrast_A_but_B_rule_invalid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(Contrast_A_but_B_rule_invalid_negative_sentiment))
        print("No of sentences with no contrast but contain A-but-B rule: ", len(No_contrast_A_but_B_rule_valid['sentence']))
        print("No of positive sentiment sentences with no contrast but contain A-but-B rule: ", len(No_contrast_A_but_B_rule_valid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(No_contrast_A_but_B_rule_valid_negative_sentiment))
        print("No of sentences with no contrast and do not contain A-but-B rule: ", len(No_contrast_A_but_B_rule_invalid['sentence']))
        print("No of positive sentiment sentences with no contrast and do not contain A-but-B rule: ", len(No_contrast_A_but_B_rule_invalid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(No_contrast_A_but_B_rule_invalid_negative_sentiment))

        print("\n")

        print("Out of all sentences containing A-but-B structure and correctly classified by VADER ({}), VADER determines that: -\n".format(vader_sentiment_corrects_of_A_but_B_structure_sentences.count(1)))
        print("No of sentences with contrast and contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_valid['sentence']))
        print("No of positive sentiment sentences with contrast and contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_valid_positive_sentiment))
        print("No of negative sentiment sentences with contrast and contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_valid_negative_sentiment))
        print("No of sentences with contrast but do not contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_invalid['sentence']))
        print("No of positive sentiment sentences with contrast but do not contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_invalid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(Correct_contrast_A_but_B_rule_invalid_negative_sentiment))
        print("No of sentences with no contrast but contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_valid['sentence']))
        print("No of positive sentiment sentences with no contrast but contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_valid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_valid_negative_sentiment))
        print("No of sentences with no contrast and do not contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_invalid['sentence']))
        print("No of positive sentiment sentences with no contrast and do not contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_invalid_positive_sentiment))
        print("No of negative sentiment sentences with contrast but do not contain A-but-B rule: ", len(Correct_no_contrast_A_but_B_rule_invalid_negative_sentiment))
        
        print("\n")

        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_1/Contrast_A_but_B_rule_valid.txt', 'w') as file:
            for index, sentence in enumerate(Contrast_A_but_B_rule_valid['sentence']):
                ground_truth_sentiment = Contrast_A_but_B_rule_valid['ground_truth_sentiment'][index]
                vader_sentiment = Contrast_A_but_B_rule_valid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Contrast_A_but_B_rule_valid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Contrast_A_but_B_rule_valid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_1/Contrast_A_but_B_rule_invalid.txt', 'w') as file:
            for index, sentence in enumerate(Contrast_A_but_B_rule_invalid['sentence']):
                ground_truth_sentiment = Contrast_A_but_B_rule_invalid['ground_truth_sentiment'][index]
                vader_sentiment = Contrast_A_but_B_rule_invalid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_1/No_contrast_A_but_B_rule_valid.txt', 'w') as file:
            for index, sentence in enumerate(No_contrast_A_but_B_rule_valid['sentence']):
                ground_truth_sentiment = No_contrast_A_but_B_rule_valid['ground_truth_sentiment'][index]
                vader_sentiment = No_contrast_A_but_B_rule_valid['vader_sentence_sentiment'][index]
                clause_A_sentiment = No_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = No_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_1/No_contrast_A_but_B_rule_invalid.txt', 'w') as file:
            for index, sentence in enumerate(No_contrast_A_but_B_rule_invalid['sentence']):
                ground_truth_sentiment = No_contrast_A_but_B_rule_invalid['ground_truth_sentiment'][index]
                vader_sentiment = No_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'][index]
                clause_A_sentiment = No_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = No_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
    


        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_2/Contrast_A_but_B_rule_valid.txt', 'w') as file:
            for index, sentence in enumerate(Correct_contrast_A_but_B_rule_valid['sentence']):
                ground_truth_sentiment = Correct_contrast_A_but_B_rule_valid['ground_truth_sentiment'][index]
                vader_sentiment = Correct_contrast_A_but_B_rule_valid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Correct_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Correct_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_2/Contrast_A_but_B_rule_invalid.txt', 'w') as file:
            for index, sentence in enumerate(Correct_contrast_A_but_B_rule_invalid['sentence']):
                ground_truth_sentiment = Correct_contrast_A_but_B_rule_invalid['ground_truth_sentiment'][index]
                vader_sentiment = Correct_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Correct_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Correct_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_2/No_contrast_A_but_B_rule_valid.txt', 'w') as file:
            for index, sentence in enumerate(Correct_no_contrast_A_but_B_rule_valid['sentence']):
                ground_truth_sentiment = Correct_no_contrast_A_but_B_rule_valid['ground_truth_sentiment'][index]
                vader_sentiment = Correct_no_contrast_A_but_B_rule_valid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Correct_no_contrast_A_but_B_rule_valid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Correct_no_contrast_A_but_B_rule_valid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))
        
        with open('datasets/SST2_sentences/preprocessed_dataset/vader_distribution/set_2/No_contrast_A_but_B_rule_invalid.txt', 'w') as file:
            for index, sentence in enumerate(Correct_no_contrast_A_but_B_rule_invalid['sentence']):
                ground_truth_sentiment = Correct_no_contrast_A_but_B_rule_invalid['ground_truth_sentiment'][index]
                vader_sentiment = Correct_no_contrast_A_but_B_rule_invalid['vader_sentence_sentiment'][index]
                clause_A_sentiment = Correct_no_contrast_A_but_B_rule_invalid['vader_clause_A_sentiment'][index]
                clause_B_sentiment = Correct_no_contrast_A_but_B_rule_invalid['vader_clause_B_sentiment'][index]
                file.write("%s GS(%d) VS(%d) A(%d) B(%d)\n" % (sentence, ground_truth_sentiment, vader_sentiment, clause_A_sentiment, clause_B_sentiment))



        with open("datasets/SST2_sentences/preprocessed_dataset/vader_distribution/vader_sentiment_scores.pickle", 'wb') as handle:
            pickle.dump(vader_sentiment_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
