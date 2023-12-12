# General imports
import os
import pickle
import ast
import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import shutil
import advertools as adv
import preprocessor as p
import re
import logging
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix
import random
import pprint
import signal
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.stderr = sys.stdout

# VADER imports
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# BERT imports
from transformers import BertTokenizer, BertModel
from transformers import logging
logging.set_verbosity_error()

# ROBERTA sentiment analysis import
# from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer
from scipy.special import softmax

# Global variables
pp = pprint.PrettyPrinter()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
# roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
# roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)




class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

class Construct_covid19_dataset(object):

    def __init__(self, config):
        self.config = config
        self.counters = None
        self.output_data = None
    
    def findWholeWord(self, w):
        
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    def tweet_preprocessor(self, input, emoji=False): # Preprocess the tweet so that identified contrastive discourse relations can applied (both manually identified and automatically identified from discourse tagger tool)
        
        if emoji == False:

            # Basic cleaning (removing URLs, hashtags, reserved words)
            p.set_options(p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION)
            cleaned_tweet = p.clean(input)

            output = cleaned_tweet
            return output
        
        elif emoji == True:
            
            # Basic cleaning (removing URLs, hashtags, reserved words)
            p.set_options(p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION)
            cleaned_tweet = p.clean(input)

            output = cleaned_tweet
            return output

    def emotion_scores(self, emojis, emo_tag1200_emoji_list, emoji_score_table):

        emotion_scores = []
        for emoji in emojis:
            if emoji in emo_tag1200_emoji_list: 
                emotion_scores_emoji = []
                emoji_score_table_row = emoji_score_table[emoji_score_table['emoji']==emoji]
                emotion_scores_emoji.append(float(emoji_score_table_row['anger']))
                emotion_scores_emoji.append(float(emoji_score_table_row['disgust']))
                emotion_scores_emoji.append(float(emoji_score_table_row['fear']))
                emotion_scores_emoji.append(float(emoji_score_table_row['joy']))
                emotion_scores_emoji.append(float(emoji_score_table_row['sadness']))
                emotion_scores_emoji.append(float(emoji_score_table_row['trust']))
                emotion_scores.append(emotion_scores_emoji)
            else:
                emotion_scores_emoji = [0.0]
                emotion_scores.append(emotion_scores_emoji)

        agg_emotions_score = 0.0
        for emoji_scores in emotion_scores:
            for score in emoji_scores:
                agg_emotions_score = agg_emotions_score + score

        return emotion_scores, agg_emotions_score
    
    def conjunction_analysis(self, sentence):

        # tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_sentence = sentence.split()

        # A-but-B
        if ('but' in tokenized_sentence and tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1): # Check if the sentence contains A-but-B structure
            
            rule_structure = "A-but-B"
            rule_conjunct = "B"

            A_clause = sentence.split('but')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('but')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause

        elif 'But' in tokenized_sentence and tokenized_sentence.index('But') != 0 and tokenized_sentence.index('But') != -1 and tokenized_sentence.count('But') == 1: # Check if the sentence contains A-but-B structure

            rule_structure = "A-but-B"
            rule_conjunct = "B"

            A_clause = sentence.split('But')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('But')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        # A-while-B
        if 'while' in tokenized_sentence and tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1:

            rule_structure = "A-while-B"
            rule_conjunct = "A"

            A_clause = sentence.split('while')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('while')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        elif 'While' in tokenized_sentence and tokenized_sentence.index('While') != 0 and tokenized_sentence.index('While') != -1 and tokenized_sentence.count('While') == 1:

            rule_structure = "A-while-B"
            rule_conjunct = "A"

            A_clause = sentence.split('While')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('While')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause

        # A-yet-B
        if 'yet' in tokenized_sentence and tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1: # Check if the sentence contains A-yet-B structure

            rule_structure = "A-yet-B"
            rule_conjunct = "B"   

            A_clause = sentence.split('yet')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('yet')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        elif 'Yet' in tokenized_sentence and tokenized_sentence.index('Yet') != 0 and tokenized_sentence.index('Yet') != -1 and tokenized_sentence.count('Yet') == 1: # Check if the sentence contains A-yet-B structure

            rule_structure = "A-yet-B"
            rule_conjunct = "B"    

            A_clause = sentence.split('Yet')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('Yet')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause

        # A-however-B
        if 'however' in tokenized_sentence and tokenized_sentence.index('however') != 0 and tokenized_sentence.index('however') != -1 and tokenized_sentence.count('however') == 1:

            rule_structure = "A-however-B"
            rule_conjunct = "B"

            A_clause = sentence.split('however')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('however')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        elif 'However' in tokenized_sentence and tokenized_sentence.index('However') != 0 and tokenized_sentence.index('However') != -1 and tokenized_sentence.count('However') == 1:

            rule_structure = "A-however-B"
            rule_conjunct = "B"

            A_clause = sentence.split('However')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('However')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        # A-despite-B
        if 'despite' in tokenized_sentence and tokenized_sentence.index('despite') != 0 and tokenized_sentence.index('despite') != -1 and tokenized_sentence.count('despite') == 1:

            rule_structure = "A-despite-B"
            rule_conjunct = "B"

            A_clause = sentence.split('despite')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('despite')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        elif 'Despite' in tokenized_sentence and tokenized_sentence.index('Despite') != 0 and tokenized_sentence.index('Despite') != -1 and tokenized_sentence.count('Despite') == 1:

            rule_structure = "A-despite-B"
            rule_conjunct = "A"

            A_clause = sentence.split('Despite')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('Despite')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        # A-although-B
        if 'although' in tokenized_sentence and tokenized_sentence.index('although') != 0 and tokenized_sentence.index('although') != -1 and tokenized_sentence.count('although') == 1:

            rule_structure = "A-although-B"
            rule_conjunct = "A"

            A_clause = sentence.split('although')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('although')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
        
        elif 'Although' in tokenized_sentence and tokenized_sentence.index('Although') != 0 and tokenized_sentence.index('Although') != -1 and tokenized_sentence.count('Although') == 1:

            rule_structure = "A-although-B"
            rule_conjunct = "A"

            A_clause = sentence.split('Although')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('Although')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause

        # A-though-B
        if 'though' in tokenized_sentence and tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1:

            rule_structure = "A-though-B"
            rule_conjunct = "A"

            A_clause = sentence.split('though')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('though')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause

        elif 'Though' in tokenized_sentence and tokenized_sentence.index('Though') != 0 and tokenized_sentence.index('Though') != -1 and tokenized_sentence.count('Though') == 1:

            rule_structure = "A-Though-B"
            rule_conjunct = "A"

            A_clause = sentence.split('Though')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = sentence.split('Though')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

            return rule_structure, rule_conjunct, A_clause, B_clause
    
        else:

            return None, None, None, None

    def covid_dataset_thresholds(self): # Code to calculate the Sentiment and VADER thresholds for the tweets

        # Emoji to emotions score table
        emoji_score_table = pd.read_csv("input_to_scripts/EmoTag1200-scores.csv")
        emoji_score_table = emoji_score_table.drop(['anger', 'disgust', 'fear', 'sadness'], axis=1)
        emoji_score_table = emoji_score_table.rename(columns={'-anger':'anger', '-disgust':'disgust', '-fear':'fear', '-sadness':'sadness'})
        emo_tag1200_emoji_list = list(emoji_score_table['emoji'])

        # Tweet hash table to check for duplicate tweets (will contain more than 1.2 mil datapoints)
        tweet_hash_table = set()

        # Thresholds
        agg_emotion_scores = []
        vader_sentence_scores = []
        vader_clause_A_scores = []
        vader_clause_B_scores = []

        # for i in range(1, 201):
        #     if i < 10:
        #         i = "0"+str(i)
        #     else:
        #         i = str(i)
            
        #     # Read the data file
        #     if os.path.exists("datasets/Covid-19_tweets/preprocessed_dataset/corona_tweets_"+i+"/preprocessed_data.pickle"):
        #         with open("datasets/Covid-19_tweets/preprocessed_dataset/corona_tweets_"+i+"/preprocessed_data.pickle", 'rb') as handle:
        #             file = pickle.load(handle)
            
        #     # Read each datapoint from the file
        #     for index, tweet_id in enumerate(file["tweet_id"]):

        #         # Print the data in the file
        #         agg_emotion_scores.append(file["agg_emotion_score"][index])
        #         vader_sentence_scores.append(file["vader_score_sentence"][index]['compound'])
        #         if file["vader_score_clause_A"][index] != 'not_applicable':
        #             vader_clause_A_scores.append(file["vader_score_clause_A"][index]['compound'])
        #         if file["vader_score_clause_B"][index] != 'not_applicable':
        #             vader_clause_B_scores.append(file["vader_score_clause_B"][index]['compound'])

        # Select a file from the corpus
        for i in range(1, 400):
            if i < 10:
                i = "0"+str(i)
            else:
                i = str(i)

            # Setup progress bar for the file
            file_name = "corona_tweets_"+i+"_data.txt"
            with open("input_to_scripts/corpus"+"/corona_tweets_"+i+"/"+file_name,'r') as inf:
                
                no_of_lines = 0
                for index, line in enumerate(inf): 
                    no_of_lines += 1

            # Select a tweet object from the file
            with open("input_to_scripts/corpus"+"/corona_tweets_"+i+"/corona_tweets_"+i+"_data.txt",'r') as inf:
                print("\n")

                # Create a progress bar
                with tqdm(desc = file_name, total = no_of_lines) as pbar:

                    for index, line in enumerate(inf):

                        # Extract relevant fields
                        tweet_dict_obj = eval(line)
                        tweet = tweet_dict_obj['text']
                        tweet_id = tweet_dict_obj['id']
                        language = tweet_dict_obj['lang']

                        # # Preprocess the tweet
                        # try:
                        #     preprocessed_tweet = self.tweet_preprocessor(tweet)
                        # except:
                        #     print(tweet)
                        #     print("Error in preprocessing the tweet" + tweet, sys.exc_info()[0])
                        #     pbar.update()
                        #     continue

                        # Extract emojis from the tweet
                        emoji_summary_dict = adv.extract_emoji([tweet])
                        emojis = emoji_summary_dict['emoji'][0]
                        emoji_names = emoji_summary_dict['emoji_text'][0]

                        # # Check if the tweet is in English
                        # if language != "en":
                        #     pbar.update()
                        #     continue

                        # # Check if the tweet contains atleast one emoji
                        # if len(emojis) == 0:
                        #     pbar.update()
                        #     continue

                        # # Check if all the emojis are present at the end of the tweet
                        # emoji_string = ''
                        # for emoji in emojis:
                        #     emoji_string = emoji_string + emoji
                        # if not preprocessed_tweet.endswith(emoji_string):
                        #     pbar.update()
                        #     continue
                        
                        # # Check if atleast one emoji is present in the emotag1200 list
                        # count_of_emojis_in_emotag1200 = 0
                        # for emoji in emojis:
                        #     if emoji in emo_tag1200_emoji_list:
                        #         count_of_emojis_in_emotag1200 += 1
                        # if count_of_emojis_in_emotag1200 == 0:
                        #     pbar.update()
                        #     continue
                            
                        # # Check if the tweet contains equal to or more than 28 chars
                        # if len(preprocessed_tweet) < 28:
                        #     pbar.update()
                        #     continue

                        # # Check if the tweet is not a duplicate tweet
                        # length_before = len(tweet_hash_table)
                        # tweet_hash_table.add(preprocessed_tweet)
                        # if length_before == len(tweet_hash_table):
                        #     pbar.update()
                        #     continue
                        
                        # Remove the emojis from the tweet
                        preprocessed_tweet = self.tweet_preprocessor(tweet, emoji=True)

                        # Calculate the emotion scores and agg emotion score
                        emotion_scores, agg_emotion_score = self.emotion_scores(emojis, emo_tag1200_emoji_list, emoji_score_table)
                        agg_emotion_scores.append(agg_emotion_score)

                        # Calculate the VADER score for the sentence
                        analyzer = SentimentIntensityAnalyzer()
                        vader_score_sentence = analyzer.polarity_scores(preprocessed_tweet)
                        vader_sentence_scores.append(vader_score_sentence['compound'])
                       
                        # Conjunction analysis
                        clause_A, clause_B, rule_strucutre, rule_conjunct = self.conjunction_analysis(preprocessed_tweet)
                        if (clause_A ==  None and clause_B == None and rule_strucutre == None and rule_conjunct == None):
                            
                            vader_score_clause_A = None
                            vader_score_clause_B = None
                            rule_strucutre = "no_structure"
                            rule_label = "not_applicable"
                            contrast = "not_applicable"

                        elif (clause_A !=  None and clause_B != None and rule_strucutre != None and rule_conjunct != None):
                            
                            # Calculate the VADER scores for tweet and its conjuncts
                            analyzer = SentimentIntensityAnalyzer() 
                            vader_score_sentence = analyzer.polarity_scores(preprocessed_tweet)
                            vader_score_clause_A = analyzer.polarity_scores(clause_A)
                            vader_score_clause_B = analyzer.polarity_scores(clause_B)
                            vader_clause_A_scores.append(vader_score_clause_A['compound'])
                            vader_clause_B_scores.append(vader_score_clause_A['compound'])

                        pbar.update()

                        if len(agg_emotion_scores) and len(vader_sentence_scores) == 1000000:
                            break
            
            if len(agg_emotion_scores) and len(vader_sentence_scores) == 1000000:
                break

        mean_agg_emotion_scores = sum(agg_emotion_scores)/len(agg_emotion_scores)
        mean_vader_sentence_scores = sum(vader_sentence_scores)/len(vader_sentence_scores)
        mean_vader_clause_A_scores = sum(vader_clause_A_scores)/len(vader_clause_A_scores)
        mean_vader_clause_B_scores = sum(vader_clause_B_scores)/len(vader_clause_B_scores)

        vairance_agg_emotion_scores = sum([((x - mean_agg_emotion_scores) ** 2) for x in agg_emotion_scores])/len(agg_emotion_scores)
        variance_vader_sentence_scores = sum([((x - mean_vader_sentence_scores) ** 2) for x in vader_sentence_scores])/len(vader_sentence_scores)
        variance_vader_clause_A_scores = sum([((x - mean_vader_clause_A_scores) ** 2) for x in vader_clause_A_scores])/len(vader_clause_A_scores)
        variance_vader_clause_B_scores = sum([((x - mean_vader_clause_B_scores) ** 2) for x in vader_clause_B_scores])/len(vader_clause_B_scores)

        stdev_agg_emotion_scores = vairance_agg_emotion_scores ** 0.5
        stdev_vader_sentence_scores = variance_vader_sentence_scores ** 0.5
        stdev_vader_clause_A_scores = variance_vader_clause_A_scores ** 0.5
        stdev_vader_clause_B_scores = variance_vader_clause_B_scores ** 0.5

        print("\n")
        print("no of agg emotion scores: ", len(agg_emotion_scores))
        print("no of vader sentence scores: ", len(vader_sentence_scores))
        print("no of vader clause A scores: ", len(vader_clause_A_scores))
        print("no of vader clause B scores: ", len(vader_clause_B_scores))
        print("mean agg emotion scores: ", mean_agg_emotion_scores)
        print("mean vader sentence scores: ", mean_vader_sentence_scores)
        print("mean vader clause A scores: ", mean_vader_clause_A_scores)
        print("mean vader clause B scores: ", mean_vader_clause_B_scores)
        print("variance agg emotion scores: ", vairance_agg_emotion_scores)
        print("variance vader sentence scores: ", variance_vader_sentence_scores)
        print("variance vader clause A scores: ", variance_vader_clause_A_scores)
        print("variance vader clause B scores: ", variance_vader_clause_B_scores)
        print("std. dev agg emotion scores: ", stdev_agg_emotion_scores)
        print("std. dev vader sentence scores: ", stdev_vader_sentence_scores)
        print("std. dev vader clause A scores: ", stdev_vader_clause_A_scores)
        print("std. dev vader clause B scores: ", stdev_vader_clause_B_scores)
        print("\n")

    def append_positive_blue_area(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if rule_structure == "no_structure" and sentiment_label == 1: # Blue area positive tweets
            if self.counters["counter_blue_area_positive"] < 300000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_blue_area_positive"] += 1
    
    def append_negative_blue_area(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if rule_structure == "no_structure" and sentiment_label == -1: # Blue area negative tweets
                        
            if self.counters["counter_blue_area_negative"] < 300000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_blue_area_negative"] += 1

    def append_no_rule_rule_syntactic_structure(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if rule_structure != "not_applicable" and rule_label == "no_rule": # No rule tweets but having syntactic structure
        
            self.output_data['tweet_id'].append(tweet_id)
            self.output_data['tweet'].append(tweet)
            self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
            self.output_data['clause_A'].append(clause_A)
            self.output_data['clause_B'].append(clause_B)
            self.output_data['emojis'].append(emojis)
            self.output_data['emoji_names'].append(emoji_names)
            self.output_data['emotion_scores'].append(emotion_scores)
            self.output_data['agg_emotion_score'].append(agg_emotion_score)
            self.output_data['sentiment_label'].append(sentiment_label)
            self.output_data['vader_score_sentence'].append(vader_score_sentence)
            self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
            self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
            self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            self.output_data['rule_structure'].append(rule_structure)
            self.output_data['rule_label'].append(rule_label)
            self.output_data['contrast'].append(contrast)

            self.counters["counter_no_rule_rule_syntactic_structure"] += 1

    def append_positive_a_but_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "contrast" and sentiment_label == 1: # A-but-B rule
            if self.counters["counter_positive_a_but_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_but_b_contrast"] += 1
    
    def append_negative_a_but_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "contrast" and sentiment_label == -1: # A-but-B rule
            if self.counters["counter_negative_a_but_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_but_b_contrast"] += 1

    def append_positive_a_but_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_but_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_but_b_no_contrast"] += 1

    def append_negative_a_but_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_but_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_but_b_no_contrast"] += 1

    def append_positive_a_yet_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_yet_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_yet_b_contrast"] += 1
    
    def append_negative_a_yet_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_yet_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_yet_b_contrast"] += 1

    def append_positive_a_yet_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_yet_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_yet_b_no_contrast"] += 1

    def append_negative_a_yet_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_yet_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_yet_b_no_contrast"] += 1

    def append_positive_a_however_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_however_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_however_b_contrast"] += 1

    def append_negative_a_however_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_however_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_however_b_contrast"] += 1

    def append_positive_a_however_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_however_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_however_b_no_contrast"] += 1

    def append_negative_a_however_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_however_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_however_b_no_contrast"] += 1

    def append_positive_a_despite_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_despite_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_despite_b_contrast"] += 1

    def append_negative_a_despite_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_despite_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_despite_b_contrast"] += 1

    def append_positive_a_despite_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_despite_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_despite_b_no_contrast"] += 1

    def append_negative_a_despite_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_despite_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_despite_b_no_contrast"] += 1

    def append_positive_a_although_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_although_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_although_b_contrast"] += 1

    def append_negative_a_although_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_although_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_although_b_contrast"] += 1

    def append_positive_a_although_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_although_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_although_b_no_contrast"] += 1

    def append_negative_a_although_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_although_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_although_b_no_contrast"] += 1

    def append_positive_a_though_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_though_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_though_b_contrast"] += 1

    def append_negative_a_though_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_though_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_though_b_contrast"] += 1

    def append_positive_a_though_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_though_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_though_b_no_contrast"] += 1

    def append_negative_a_though_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_though_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_though_b_no_contrast"] += 1

    def append_positive_a_while_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_while_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_while_b_contrast"] += 1

    def append_negative_a_while_b_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
        
        if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_while_b_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_while_b_contrast"] += 1

    def append_positive_a_while_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "no_contrast" and sentiment_label == 1:
            if self.counters["counter_positive_a_while_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_positive_a_while_b_no_contrast"] += 1

    def append_negative_a_while_b_no_contrast(self, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

        if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "no_contrast" and sentiment_label == -1:
            if self.counters["counter_negative_a_while_b_no_contrast"] < 25000:
                self.output_data['tweet_id'].append(tweet_id)
                self.output_data['tweet'].append(tweet)
                self.output_data['preprocessed_tweet'].append(preprocessed_tweet)
                self.output_data['clause_A'].append(clause_A)
                self.output_data['clause_B'].append(clause_B)
                self.output_data['emojis'].append(emojis)
                self.output_data['emoji_names'].append(emoji_names)
                self.output_data['emotion_scores'].append(emotion_scores)
                self.output_data['agg_emotion_score'].append(agg_emotion_score)
                self.output_data['sentiment_label'].append(sentiment_label)
                self.output_data['vader_score_sentence'].append(vader_score_sentence)
                self.output_data['vader_score_clause_A'].append(vader_score_clause_A)
                self.output_data['vader_score_clause_B'].append(vader_score_clause_B)
                self.output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
                self.output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
                self.output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
                self.output_data['rule_structure'].append(rule_structure)
                self.output_data['rule_label'].append(rule_label)
                self.output_data['contrast'].append(contrast)

                self.counters["counter_negative_a_while_b_no_contrast"] += 1

    def covid_dataset(self, process_no, start, stop): # Code to label the tweets with sentiment labels

        # Emoji to emotions score table
        emoji_score_table = pd.read_csv("datasets/EmoTag1200-scores.csv")
        emoji_score_table = emoji_score_table.drop(['anger', 'disgust', 'fear', 'sadness'], axis=1)
        emoji_score_table = emoji_score_table.rename(columns={'-anger':'anger', '-disgust':'disgust', '-fear':'fear', '-sadness':'sadness'})
        emo_tag1200_emoji_list = list(emoji_score_table['emoji'])

        # Tweets hash table (to check for duplicate tweets)(WRONG since it's only checking tweets for a particular file)
        tweets_hash_table = set()

        # Select a file from the corpus
        for i in range(start, stop+1):
            i = str(i)

            # Output data
            self.output_data = {
                                'tweet_id':[],
                                'tweet':[],
                                'preprocessed_tweet':[],
                                'clause_A':[],
                                'clause_B':[],
                                'emojis':[], 
                                'emoji_names':[],
                                'emotion_scores':[],
                                'agg_emotion_score':[],
                                'sentiment_label':[],
                                'vader_score_sentence':[],
                                'vader_score_clause_A':[],
                                'vader_score_clause_B':[],
                                'vader_sentiment_sentence':[],
                                'vader_sentiment_clause_A':[],
                                'vader_sentiment_clause_B':[],
                                'rule_structure':[],
                                'rule_label':[],
                                'contrast':[]
                                }
            
            # Counters for the output data
            self.counters = {
                            "tweets_in_corpus":0,
                            "tweets_in_distribution":0,

                            "consistency_check":[],
                            "vader_accuracy":[],
                            "roberta_accuracy":[],

                            "counter_blue_area_positive":0,
                            "counter_blue_area_negative":0,

                            "counter_no_rule_rule_syntactic_structure":0,

                            "counter_positive_a_but_b_contrast":0,
                            "counter_negative_a_but_b_contrast":0,
                            "counter_positive_a_but_b_no_contrast":0,
                            "counter_negative_a_but_b_no_contrast":0,

                            "counter_positive_a_yet_b_contrast":0,
                            "counter_negative_a_yet_b_contrast":0,
                            "counter_positive_a_yet_b_no_contrast":0,
                            "counter_negative_a_yet_b_no_contrast":0,

                            "counter_positive_a_however_b_contrast":0,
                            "counter_negative_a_however_b_contrast":0,
                            "counter_positive_a_however_b_no_contrast":0,
                            "counter_negative_a_however_b_no_contrast":0,

                            "counter_positive_a_despite_b_contrast":0,
                            "counter_negative_a_despite_b_contrast":0,
                            "counter_positive_a_despite_b_no_contrast":0,
                            "counter_negative_a_despite_b_no_contrast":0,

                            "counter_positive_a_although_b_contrast":0,
                            "counter_negative_a_although_b_contrast":0,
                            "counter_positive_a_although_b_no_contrast":0,
                            "counter_negative_a_although_b_no_contrast":0,

                            "counter_positive_a_though_b_contrast":0,
                            "counter_negative_a_though_b_contrast":0,
                            "counter_positive_a_though_b_no_contrast":0,
                            "counter_negative_a_though_b_no_contrast":0,

                            "counter_positive_a_while_b_contrast":0,
                            "counter_negative_a_while_b_contrast":0,
                            "counter_positive_a_while_b_no_contrast":0,
                            "counter_negative_a_while_b_no_contrast":0,
                            }
        
            # Make directories for output data
            if not os.path.exists("datasets/corpus/corona_tweets_"+i):
                os.makedirs("datasets/corpus/corona_tweets_"+i)

            # Setup progress bar for the file
            file_name = "corona_tweets_"+i+"_data.txt"
            with open("datasets/corpus"+"/corona_tweets_"+i+"/"+file_name,'r') as inf:
                no_of_lines = 0
                for line in inf: 
                    no_of_lines += 1

            # Create log file
            log_file_name = "preprocessed_data_log.log"
            old_stdout = sys.stdout
            log_file = open("datasets/corpus/corona_tweets_"+i+"/"+log_file_name, "w")

            # Tweet index in the file
            tweet_index = 0

            # Select a tweet object from the file
            with open("datasets/corpus"+"/corona_tweets_"+i+"/corona_tweets_"+i+"_data.txt","r") as inf:
                print("\n")

                # Create a progress bar
                with tqdm(desc = file_name, total = no_of_lines) as pbar:

                    # Read the tweet
                    for line in inf:
                        
                        # Start the timeout timer
                        signal.alarm(600)

                        try:
                            tweet_index += 1
                            sys.stdout = log_file
                        
                            # Update tweets in corpus
                            self.counters["tweets_in_corpus"] += 1

                            # Extract relevant fields
                            try:
                                tweet_dict_obj = eval(line)
                                tweet = tweet_dict_obj['text']
                                tweet_id = tweet_dict_obj['id']
                                language = tweet_dict_obj['lang']
                            except:
                                continue

                            # Preprocess the tweet (remove hashtages, URLs, reserved keywords, @mentions and spaces)
                            try:
                                preprocessed_tweet = self.tweet_preprocessor(tweet)
                            except:
                                print(tweet)
                                print("Error in preprocessing the tweet" + tweet, sys.exc_info()[0])
                                pbar.update()
                                continue

                            # Extract emojis from the tweet
                            try:
                                emoji_summary_dict = adv.extract_emoji([preprocessed_tweet])
                                emojis = emoji_summary_dict['emoji'][0]
                                emoji_names = emoji_summary_dict['emoji_text'][0]
                            except:
                                continue

                            # Check if the tweet is in English
                            if language != "en":
                                pbar.update()
                                continue

                            # Check if the tweet contains atleast one emoji
                            if len(emojis) == 0:
                                pbar.update()
                                continue

                            # Check if all the emojis are present at the end of the tweet
                            emoji_string = ''
                            for emoji in emojis:
                                emoji_string = emoji_string + emoji
                            if not preprocessed_tweet.endswith(emoji_string):
                                pbar.update()
                                continue
                            
                            # Check if atleast one emoji is present in the emotag1200 list
                            count_of_emojis_in_emotag1200 = 0
                            for emoji in emojis:
                                if emoji in emo_tag1200_emoji_list:
                                    count_of_emojis_in_emotag1200 += 1
                            if count_of_emojis_in_emotag1200 == 0:
                                pbar.update()
                                continue
                                
                            # Check if the tweet contains equal to or more than 28 chars
                            if len(preprocessed_tweet) < 28:
                                pbar.update()
                                continue
                                
                            # Check if the tweet is not a duplicate tweet
                            length_before = len(tweets_hash_table)
                            tweets_hash_table.add(preprocessed_tweet)
                            if length_before == len(tweets_hash_table):
                                pbar.update()
                                continue
                            
                            # Remove the emojis from the tweet
                            preprocessed_tweet = self.tweet_preprocessor(preprocessed_tweet, emoji=True)

                            # Calculate the emotion scores and agg emotion score
                            emotion_scores, agg_emotion_score = self.emotion_scores(emojis, emo_tag1200_emoji_list, emoji_score_table)
                            
                            # Assign sentiment labels
                            sentiment_label = 0
                            if agg_emotion_score > 2.83:
                                sentiment_label = 1
                            elif agg_emotion_score < -2.83:
                                sentiment_label = -1
                            else:
                                pbar.update()
                                continue                        

                            # Calculate the VADER score and sentiment label for the sentence
                            try:
                                analyzer = SentimentIntensityAnalyzer()
                                vader_score_sentence = analyzer.polarity_scores(preprocessed_tweet)
                                if vader_score_sentence['compound'] >= 0.05:
                                    vader_sentiment_sentence = 1 # Positive sentiment
                                elif (vader_score_sentence['compound'] > -0.05) and (vader_score_sentence['compound'] < 0.05):
                                    vader_sentiment_sentence = 0 # Neutral sentiment
                                elif vader_score_sentence["compound"] <= -0.05:
                                    vader_sentiment_sentence = -1 # Negative sentiment
                            except:
                                continue

                            # # Calculate the ROBERTA sentiment label for the sentence
                            # try:
                            #     roberta_input = roberta_tokenizer(preprocessed_tweet, return_tensors='pt')
                            #     roberta_output = roberta_model(**roberta_input)
                            #     roberta_scores = roberta_output[0][0].detach().numpy()
                            #     softmax_scores = softmax(roberta_scores)
                            #     ranking = np.argsort(softmax_scores)
                            #     ranking = ranking[::-1]
                            #     roberta_sentiment_label = ranking[0]
                            #     if roberta_sentiment_label == 0:
                            #         roberta_sentiment_label = -1
                            #     elif roberta_sentiment_label == 1:
                            #         roberta_sentiment_label = 0
                            #     elif roberta_sentiment_label == 2:
                            #         roberta_sentiment_label = 1
                            # except:
                            #     continue

                            # VADER consistency check
                            if vader_sentiment_sentence != sentiment_label:
                                self.counters["vader_accuracy"].append(0)
                                self.counters["consistency_check"].append(0)
                                pbar.update()
                                continue
                            elif vader_sentiment_sentence == sentiment_label:
                                self.counters["vader_accuracy"].append(1)
                                self.counters["consistency_check"].append(1)

                            # # ROBERTA consistency check
                            # if roberta_sentiment_label != sentiment_label:
                            #     self.counters["roberta_accuracy"].append(0)
                            #     self.counters["consistency_check"].append(0)
                            #     pbar.update()
                            #     continue
                            # elif vader_sentiment_sentence == sentiment_label:
                            #     self.counters["roberta_accuracy"].append(1)
                            #     self.counters["consistency_check"].append(1)

                            # # VADER and ROBERTA consistency check
                            # if vader_sentiment_sentence != sentiment_label and roberta_sentiment_label != sentiment_label:
                            #     self.counters["consistency_check"].append(0)
                            #     self.counters["vader_accuracy"].append(0)
                            #     self.counters["roberta_accuracy"].append(0)
                            #     pbar.update()
                            #     continue
                            # elif vader_sentiment_sentence == sentiment_label and roberta_sentiment_label != sentiment_label:
                            #     self.counters["consistency_check"].append(1)
                            #     self.counters["vader_accuracy"].append(1)
                            #     self.counters["roberta_accuracy"].append(0)
                            # elif vader_sentiment_sentence != sentiment_label and roberta_sentiment_label == sentiment_label:
                            #     self.counters["consistency_check"].append(1)
                            #     self.counters["vader_accuracy"].append(0)
                            #     self.counters["roberta_accuracy"].append(1)

                            # Conjunction analysis
                            try:
                                rule_structure, rule_conjunct, clause_A, clause_B = self.conjunction_analysis(preprocessed_tweet)
                            except:
                                continue

                            if (clause_A ==  None and clause_B == None and rule_structure == None and rule_conjunct == None):
                                
                                clause_A = "none"
                                clause_B = "none"
                                vader_score_clause_A = "not_applicable"
                                vader_score_clause_B = "not_applicable"
                                vader_sentiment_clause_A = "not_applicable"
                                vader_sentiment_clause_B = "not_applicable"
                                rule_structure = "no_structure"
                                rule_label = "not_applicable"
                                contrast = "not_applicable"

                            elif (clause_A !=  None and clause_B != None and rule_structure != None and rule_conjunct != None):
                                
                                # Calculate the VADER scores for tweet and its conjuncts
                                try:
                                    analyzer = SentimentIntensityAnalyzer() 
                                    vader_score_sentence = analyzer.polarity_scores(preprocessed_tweet)
                                    vader_score_clause_A = analyzer.polarity_scores(clause_A)
                                    vader_score_clause_B = analyzer.polarity_scores(clause_B)
                                except:
                                    continue

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
                                
                                # Check the rule is applicable
                                if rule_conjunct == "A":
                                    if vader_sentiment_clause_A == sentiment_label:
                                        rule_label = rule_structure
                                        if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                            contrast = "contrast"
                                        elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                            contrast = "no_contrast"
                                    else:
                                        rule_label = "no_rule"
                                        if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                            contrast = "contrast"
                                        elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                            contrast = "no_contrast"
                                
                                elif rule_conjunct == "B":
                                    if vader_sentiment_clause_B == sentiment_label:
                                        rule_label = rule_structure
                                        if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                            contrast = "contrast"
                                        elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                            contrast = "no_contrast"
                                    else:
                                        rule_label = "no_rule"
                                        if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                            contrast = "contrast"
                                        elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                            contrast = "no_contrast"

                            # Print the values and create their log
                            print("\n")
                            print("tweet index in the file: ", tweet_index)
                            print("tweet id: ", tweet_id)
                            print("tweet: ", tweet)
                            print("preprocessed tweet: ", preprocessed_tweet)
                            print("clause A: ", clause_A)
                            print("clause B: ", clause_B)
                            print("emojis: ", emojis)
                            print("emoji names: ", emoji_names)
                            print("emotion scores: ", emotion_scores)
                            print("agg emotion score: ", agg_emotion_score)
                            print("sentiment label: ", sentiment_label)
                            print("vader score sentence: ", vader_score_sentence)
                            print("vader score clause A: ", vader_score_clause_A)
                            print("vader score clause B: ", vader_score_clause_B)
                            print("vader sentiment: ", vader_sentiment_sentence)
                            print("vader sentiment clause A: ", vader_sentiment_clause_A)
                            print("vader sentiment clause B: ", vader_sentiment_clause_B)
                            print("rule structure: ", rule_structure)
                            print("rule label: ", rule_label)
                            print("contrast: ", contrast)
                            print("\n")

                            # Append the tweet to the output data and update the counters (only append should occur for each tweet)
                            self.append_positive_blue_area(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_blue_area(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            
                            self.append_no_rule_rule_syntactic_structure(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            
                            self.append_positive_a_but_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_but_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_but_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_but_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            
                            self.append_positive_a_while_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_while_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_while_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_while_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)

                            self.append_positive_a_yet_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_yet_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_yet_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_yet_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)

                            self.append_positive_a_however_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_however_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_however_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_however_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)

                            self.append_positive_a_despite_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_despite_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_despite_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_despite_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)

                            self.append_positive_a_although_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_although_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_although_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_although_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)

                            self.append_positive_a_though_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_though_b_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_positive_a_though_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            self.append_negative_a_though_b_no_contrast(tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast)
                            
                            # Update the tweets in distribution counter after append
                            self.counters["tweets_in_distribution"] += 1

                            print("total number of tweets processed from the corpus: ", self.counters["tweets_in_corpus"])
                            print("total number of tweets in distribution: ", self.counters["tweets_in_distribution"])
                            print("\n")

                            # Save all the counters
                            if not os.path.exists("datasets/covid19-twitter/preprocessed_dataset"):
                                os.makedirs("datasets/covid19-twitter/preprocessed_dataset")
                            with open("datasets/covid19-twitter/preprocessed_dataset/counters_"+i+".pickle", 'wb') as handle:
                                pickle.dump(self.counters, handle, protocol=pickle.HIGHEST_PROTOCOL)

                            # # Apply the break conditions on the counter
                            # if self.counters["counter_blue_area_positive"] == 300000 and self.counters["counter_blue_area_positive"] == 300000 and (self.counters["counter_positive_a_but_b_contrast"] == 25000 and self.counters["counter_negative_a_but_b_contrast"] == 25000 and self.counters["counter_positive_a_but_b_no_contrast"] == 25000 and self.counters["counter_negative_a_but_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_yet_b_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_contrast"] == 25000 and self.counters["counter_positive_a_yet_b_no_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_however_b_contrast"] == 25000 and self.counters["counter_negative_a_however_b_contrast"] == 25000 and self.counters["counter_positive_a_however_b_no_contrast"] == 25000 and self.counters["counter_negative_a_however_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_while_b_contrast"] == 25000 and self.counters["counter_negative_a_while_b_contrast"] == 25000 and self.counters["counter_positive_a_while_b_no_contrast"] == 25000 and self.counters["counter_negative_a_while_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_despite_b_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_contrast"] == 25000 and self.counters["counter_positive_a_despite_b_no_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_though_b_contrast"] == 25000 and self.counters["counter_negative_a_though_b_contrast"] == 25000 and self.counters["counter_positive_a_though_b_no_contrast"] == 25000 and self.counters["counter_negative_a_though_b_no_contrast"] == 25000):
                            #     break

                            pbar.update()
                        
                        except TimeoutException:
                            pbar.update()
                            continue
                        
                        else:
                            signal.alarm(0)

            sys.stdout.close()
            sys.stdout = old_stdout

            # Print all the counters
            print("\n")
            print("Total number of tweets processed from the corpus: ", self.counters["tweets_in_corpus"])
            print("Total number of tweets in distribution: ", self.counters["tweets_in_distribution"])
            print("\n")
            print("Tweets before consistency check: ", len(self.counters["consistency_check"]))
            print("Tweets after consistency check: ", self.counters["consistency_check"].count(1))
            print("No of tweets filtered by consistency check: ", sum(self.counters["consistency_check"])/len(self.counters["consistency_check"]))
            print("\n")
            print("Blue area, positive tweets: ", self.counters["counter_blue_area_positive"])
            print("Blue area negative tweets: ", self.counters["counter_blue_area_negative"])
            print("\n")
            print("Contain rule structure but corresponding rule is not applicable as per its linguistic definition: ", self.counters["counter_no_rule_rule_syntactic_structure"])
            print("\n")
            print("A-but-B rule, contrast and Positive: ", self.counters["counter_positive_a_but_b_contrast"])
            print("A-but-B rule, contrast and Negative: ", self.counters["counter_negative_a_but_b_contrast"])
            print("A-but-B rule, no contrast and Positive: ", self.counters["counter_positive_a_but_b_no_contrast"])
            print("A-but-B rule, no contrast and Negative: ", self.counters["counter_negative_a_but_b_no_contrast"])
            print("\n")
            print("A-yet-B rule, contrast and Positive: ", self.counters["counter_positive_a_yet_b_contrast"])
            print("A-yet-B rule, contrast and Negative: ", self.counters["counter_negative_a_yet_b_contrast"])
            print("A-yet-B rule, no contrast and Positive: ", self.counters["counter_positive_a_yet_b_no_contrast"])
            print("A-yet-B rule, no contrast and Negative: ", self.counters["counter_negative_a_yet_b_no_contrast"])
            print("\n")
            print("A-however-B rule, contrast and Positive: ", self.counters["counter_positive_a_however_b_contrast"])
            print("A-however-B rule, contrast and Negative: ", self.counters["counter_negative_a_however_b_contrast"])
            print("A-however-B rule, no contrast and Positive: ", self.counters["counter_positive_a_however_b_no_contrast"])
            print("A-however-B rule, no contrast and Negative: ", self.counters["counter_negative_a_however_b_no_contrast"])
            print("\n")
            print("A-despite-B rule, contrast and Positive: ", self.counters["counter_positive_a_despite_b_contrast"])
            print("A-despite-B rule, contrast and Negative: ", self.counters["counter_negative_a_despite_b_contrast"])
            print("A-despite-B rule, no contrast and Positive: ", self.counters["counter_positive_a_despite_b_no_contrast"])
            print("A-despite-B rule, no contrast and Negative: ", self.counters["counter_negative_a_despite_b_no_contrast"])
            print("\n")
            print("A-although-B rule, contrast and Positive: ", self.counters["counter_positive_a_although_b_contrast"])
            print("A-although-B rule, contrast and Negative: ", self.counters["counter_negative_a_although_b_contrast"])
            print("A-although-B rule, no contrast and Positive: ", self.counters["counter_positive_a_although_b_no_contrast"])
            print("A-although-B rule, no contrast and Negative: ", self.counters["counter_negative_a_although_b_no_contrast"])
            print("\n")
            print("A-though-B rule, contrast and Positive: ", self.counters["counter_positive_a_though_b_contrast"])
            print("A-though-B rule, contrast and Negative: ", self.counters["counter_negative_a_though_b_contrast"])
            print("A-though-B rule, no contrast and Positive: ", self.counters["counter_positive_a_though_b_no_contrast"])
            print("A-though-B rule, no contrast and Negative: ", self.counters["counter_negative_a_though_b_no_contrast"])
            print("\n")
            print("A-while-B rule, contrast and Positive: ", self.counters["counter_positive_a_while_b_contrast"])
            print("A-while-B rule, contrast and Negative: ", self.counters["counter_negative_a_while_b_contrast"])
            print("A-while-B rule, no contrast and Positive: ", self.counters["counter_positive_a_while_b_no_contrast"])
            print("A-while-B rule, no contrast and Negative: ", self.counters["counter_negative_a_while_b_no_contrast"])
            print("\n")

            # Save the output data
            with open("datasets/covid19-twitter/preprocessed_dataset/corona_tweets_"+i+"/preprocessed_data.pickle", 'wb') as handle:
                pickle.dump(self.output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # # Apply the break conditions on the counter
            # if self.counters["counter_blue_area_positive"] == 300000 and self.counters["counter_blue_area_positive"] == 300000 and (self.counters["counter_positive_a_but_b_contrast"] == 25000 and self.counters["counter_negative_a_but_b_contrast"] == 25000 and self.counters["counter_positive_a_but_b_no_contrast"] == 25000 and self.counters["counter_negative_a_but_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_yet_b_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_contrast"] == 25000 and self.counters["counter_positive_a_yet_b_no_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_however_b_contrast"] == 25000 and self.counters["counter_negative_a_however_b_contrast"] == 25000 and self.counters["counter_positive_a_however_b_no_contrast"] == 25000 and self.counters["counter_negative_a_however_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_while_b_contrast"] == 25000 and self.counters["counter_negative_a_while_b_contrast"] == 25000 and self.counters["counter_positive_a_while_b_no_contrast"] == 25000 and self.counters["counter_negative_a_while_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_despite_b_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_contrast"] == 25000 and self.counters["counter_positive_a_despite_b_no_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_though_b_contrast"] == 25000 and self.counters["counter_negative_a_though_b_contrast"] == 25000 and self.counters["counter_positive_a_though_b_no_contrast"] == 25000 and self.counters["counter_negative_a_though_b_no_contrast"] == 25000):
            #     break
                        
if __name__=='__main__':

    #gather parser arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_no",
                        type=int,
                        required=True)
    parser.add_argument("--start",
                        type=int,
                        required=True)
    parser.add_argument("--stop",
                        type=int,
                        required=True)
    args = parser.parse_args()
    config = vars(args)
    print("\n")
    pprint.pprint(config)
    print("\n")

    Construct_covid19_dataset(config).covid_dataset(config["process_no"],
                                                    config["start"],
                                                    config["stop"])
                                





                        
                    
                    



            

        