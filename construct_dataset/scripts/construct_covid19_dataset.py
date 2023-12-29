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
from datetime import datetime
import traceback
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

from scripts.construct_dataset.conjunction_analysis import conjunction_analysis
# from scripts.construct_dataset.append import *
from scripts.construct_dataset.emoji_thresholds import emoji_thresholds

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
            p.set_options(p.OPT.EMOJI, p.OPT.SMILEY)
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

    def create_dataset(self): # Code to label the tweets with sentiment labels

        # Emoji to emotions score table
        emoji_score_table = pd.read_csv("datasets/EmoTag1200-scores.csv")
        emoji_score_table = emoji_score_table.drop(['anger', 'disgust', 'fear', 'sadness'], axis=1)
        emoji_score_table = emoji_score_table.rename(columns={'-anger':'anger', '-disgust':'disgust', '-fear':'fear', '-sadness':'sadness'})
        emo_tag1200_emoji_list = list(emoji_score_table['emoji'])

        # Tweets hash table (to check for duplicate tweets)(WRONG since it's only checking tweets for a particular file)
        tweets_hash_table = set()

        # Select a file from the corpus
        # for i in range(self.config["start"], self.config["stop"]+1):
        #     i = str(i)

        # Output data
        self.output_data = {
                            'tweet_id':[],
                            'tweet':[],
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

                        "counter_positive_a_nevertheless_b_contrast":0,
                        "counter_negative_a_nevertheless_b_contrast":0,
                        "counter_positive_a_nevertheless_b_no_contrast":0,
                        "counter_negative_a_nevertheless_b_no_contrast":0,

                        "counter_positive_a_otherwise_b_contrast":0,
                        "counter_negative_a_otherwise_b_contrast":0,
                        "counter_positive_a_otherwise_b_no_contrast":0,
                        "counter_negative_a_otherwise_b_no_contrast":0,

                        "counter_positive_a_still_b_contrast":0,
                        "counter_negative_a_still_b_contrast":0,
                        "counter_positive_a_still_b_no_contrast":0,
                        "counter_negative_a_still_b_no_contrast":0,

                        "counter_positive_a_till_b_contrast":0,
                        "counter_negative_a_till_b_contrast":0,
                        "counter_positive_a_till_b_no_contrast":0,
                        "counter_negative_a_till_b_no_contrast":0,

                        "counter_positive_a_until_b_contrast":0,
                        "counter_negative_a_until_b_contrast":0,
                        "counter_positive_a_until_b_no_contrast":0,
                        "counter_negative_a_until_b_no_contrast":0,

                        "counter_positive_a_in spite_b_contrast":0,
                        "counter_negative_a_in spite_b_contrast":0,
                        "counter_positive_a_in spite_b_no_contrast":0,
                        "counter_negative_a_in spite_b_no_contrast":0,

                        "counter_positive_a_nonetheless_b_contrast":0,
                        "counter_negative_a_nonetheless_b_contrast":0,
                        "counter_positive_a_nonetheless_b_no_contrast":0,
                        "counter_negative_a_nonetheless_b_no_contrast":0,
                        }

        # Make directories for output data
        if not os.path.exists("datasets/corpus/corona_tweets_"+self.config["folder_number"]):
            os.makedirs("datasets/corpus/corona_tweets_"+self.config["folder_number"])
        
        # # Create log file
        # log_file_name = "preprocessed_data_log.log"
        # old_stdout = sys.stdout
        # log_file = open("datasets/corpus/corona_tweets_"+self.config["folder_number"]+"/"+log_file_name, "w")
        old_stdout = sys.stdout
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if not os.path.exists("assets/logs/corona_tweets_"+self.config["folder_number"]+"/"):
            os.makedirs("assets/logs/corona_tweets_"+self.config["folder_number"]+"/")
        log_file = open("assets/logs/corona_tweets_"+self.config["folder_number"]+"/"+self.config["asset_name"]+"_"+current_time+".txt","w")
        sys.stdout = log_file

        # Setup progress bar for the file
        file_name = "corona_tweets_"+self.config["folder_number"]+"_data.txt"
        try:
            with open("datasets/corpus"+"/corona_tweets_"+self.config["folder_number"]+"/"+file_name,'r') as inf:
                no_of_lines = 0
                for line in inf: 
                    no_of_lines += 1
        except:
            traceback.print_exc()
            print("\n"+file_name+" does not exist")
            sys.exit()

        # Tweet index in the file
        tweet_index = 0

        # Select a tweet object from the file
        with open("datasets/corpus"+"/corona_tweets_"+self.config["folder_number"]+"/corona_tweets_"+self.config["folder_number"]+"_data.txt","r") as inf:
            print("\n")

            # Create a progress bar
            with tqdm(desc = file_name, total = no_of_lines) as pbar:

                # Read the tweet
                for line in inf:
                    
                    # Start the timeout timer
                    signal.alarm(600)

                    try:
                        tweet_index += 1
                    
                        # Update tweets in corpus
                        self.counters["tweets_in_corpus"] += 1

                        # Extract relevant fields
                        try:
                            tweet_dict_obj = eval(line)
                            tweet = tweet_dict_obj['text']
                            tweet_id = tweet_dict_obj['id']
                            language = tweet_dict_obj['lang']
                        except:
                            # print("\n"+line)
                            # print("\nCould not extract relevant fields")
                            continue

                        # # Preprocess the tweet (remove hashtages, URLs, reserved keywords, @mentions and spaces)
                        # try:
                        #     preprocessed_tweet = self.tweet_preprocessor(tweet)
                        # except:
                        #     print(tweet)
                        #     print("Error in preprocessing the tweet" + tweet, sys.exc_info()[0])
                        #     pbar.update()
                        #     continue

                        # Extract emojis from the tweet
                        try:
                            emoji_summary_dict = adv.extract_emoji([tweet])
                            emojis = emoji_summary_dict['emoji'][0]
                            emoji_names = emoji_summary_dict['emoji_text'][0]
                        except:
                            # print("\n"+line)
                            # print("\nCould not extract emojis")
                            continue

                        # Check if the tweet is in English
                        if language != "en":
                            # print("\n"+line)
                            # print("\nTweet is not in English")
                            pbar.update()
                            continue

                        # Check if the tweet contains atleast one emoji
                        if len(emojis) == 0:
                            # print("\n"+line)
                            # print("\nTweet does not contains any emojis")
                            pbar.update()
                            continue

                        # Check if all the emojis are present at the end of the tweet
                        emoji_string = ''
                        for emoji in emojis:
                            emoji_string = emoji_string + emoji
                        if not tweet.endswith(emoji_string):
                            # print("\n"+line)
                            # print("\nTweet does not ends with emojis")
                            pbar.update()
                            continue
                        
                        # Check if atleast one emoji is present in the emotag1200 list
                        count_of_emojis_in_emotag1200 = 0
                        for emoji in emojis:
                            if emoji in emo_tag1200_emoji_list:
                                count_of_emojis_in_emotag1200 += 1
                        if count_of_emojis_in_emotag1200 == 0:
                            # print("\n"+line)
                            # print("\nTweet does not have any emojis in emotag1200 table")
                            pbar.update()
                            continue
                            
                        # Check if the tweet contains equal to or more than 28 chars (SPAM check)
                        if len(tweet) < 28:
                            print("\n"+line)
                            print("\nTweet contains less than 28 characters")
                            pbar.update()
                            continue
                            
                        # # Check if the tweet is not a duplicate tweet 
                        # length_before = len(tweets_hash_table)
                        # tweets_hash_table.add(tweet)
                        # if length_before == len(tweets_hash_table):
                        #     print("\n"+line)
                        #     print("\nTweet is a duplicate and has been already processed")
                        #     pbar.update()
                        #     continue

                        # Check if the tweet is a Retweet
                        # tweet_lowercase = tweet.lower()
                        # if tweet_lowercase.startswith("rt @") == True:
                        #     print("\n"+line)
                        #     print("\nTweet is a retweet")
                        #     pbar.update()
                        #     continue
                        
                        # Remove the emojis from the tweet
                        tweet_without_emojis = self.tweet_preprocessor(tweet, emoji=True)

                        # Calculate the emotion scores and agg emotion score
                        emotion_scores, agg_emotion_score = self.emotion_scores(emojis, emo_tag1200_emoji_list, emoji_score_table)
                        
                        # Assign sentiment labels
                        sentiment_label = 0
                        if agg_emotion_score > 2.83:
                            sentiment_label = 1
                        elif agg_emotion_score < -2.83:
                            sentiment_label = -1
                        else:
                            # print("\n"+line)
                            # print("\nTweet agg. emotion score is between -2.83 to 2.83")
                            pbar.update()
                            continue                        

                        # Calculate the VADER score and sentiment label for the sentence
                        try:
                            analyzer = SentimentIntensityAnalyzer()
                            # vader_score_sentence = analyzer.polarity_scores(preprocessed_tweet)
                            vader_score_sentence = analyzer.polarity_scores(tweet_without_emojis)
                            if vader_score_sentence['compound'] >= 0.05:
                                vader_sentiment_sentence = 1 # Positive sentiment
                            elif (vader_score_sentence['compound'] > -0.05) and (vader_score_sentence['compound'] < 0.05):
                                vader_sentiment_sentence = 0 # Neutral sentiment
                            elif vader_score_sentence["compound"] <= -0.05:
                                vader_sentiment_sentence = -1 # Negative sentiment
                        except:
                            # print("\n"+line)
                            # print("\nCould not calculate VADER score for the tweet")
                            continue

                        # # Calculate the ROBERTA sentiment label for the sentence
                        # try:
                        #     roberta_input = roberta_tokenizer(return_tensors='pt')
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
                            # print("\n"+line)
                            # print("\nVADER sentiment label is not equal to emoji sentiment label")
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
                            rule_structure, rule_conjunct, clause_A, clause_B = conjunction_analysis(tweet_without_emojis)
                        except:
                            print("\n"+line)
                            print("\nCould not perform conjunction analysis on the tweet")
                            traceback.print_exc()
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
                                vader_score_sentence = analyzer.polarity_scores(tweet_without_emojis)
                                vader_score_clause_A = analyzer.polarity_scores(clause_A)
                                vader_score_clause_B = analyzer.polarity_scores(clause_B)
                            except:
                                print("\n"+line)
                                print("\nCould not calculate VADER scores for the tweet conjuncts")
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
                            
                            # # Check the rule is applicable (old condition)
                            # if rule_conjunct == "A":
                            #     if vader_sentiment_clause_A == sentiment_label:
                            #         rule_label = rule_structure
                            #         if vader_sentiment_clause_A != vader_sentiment_clause_B:
                            #             contrast = "contrast"
                            #         elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                            #             contrast = "no_contrast"
                            #     else:
                            #         #condition 1 (old dataset)
                            #         # rule_label = "no_rule"
                            #         # if vader_sentiment_clause_A != vader_sentiment_clause_B:
                            #         #     contrast = "contrast"
                            #         # elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                            #         #     contrast = "no_contrast"

                            #         #condition 2 (in the published version)
                            #         continue
                            
                            # elif rule_conjunct == "B":
                            #     if vader_sentiment_clause_B == sentiment_label:
                            #         rule_label = rule_structure
                            #         if vader_sentiment_clause_A != vader_sentiment_clause_B:
                            #             contrast = "contrast"
                            #         elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                            #             contrast = "no_contrast"
                            #     else:
                            #         #condition 1 (old dataset)
                            #         # rule_label = "no_rule"
                            #         # if vader_sentiment_clause_A != vader_sentiment_clause_B:
                            #         #     contrast = "contrast"
                            #         # elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                            #         #     contrast = "no_contrast"
                                    
                            #         #condition 2 (in the published version)
                            #         continue
                            
                            # Check the rule is applicable (new condition)
                            if rule_conjunct == "a":
                                rule_label = rule_structure
                                if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                    contrast = "contrast"
                                elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                    contrast = "no_contrast"
                            
                            elif rule_conjunct == "b":
                                rule_label = rule_structure
                                if vader_sentiment_clause_A != vader_sentiment_clause_B:
                                    contrast = "contrast"
                                elif vader_sentiment_clause_A == vader_sentiment_clause_B:
                                    contrast = "no_contrast"

                        # Print the values and create their log
                        print("\n")
                        print("tweet index in the file: ", tweet_index)
                        print("tweet id: ", tweet_id)
                        print("tweet: ", tweet)
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

                        # Append to the output data
                        self.output_data['tweet_id'].append(tweet_id)
                        self.output_data['tweet'].append(tweet)
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

                        # Update the counters
                        self.counters["tweets_in_distribution"] += 1
                        if rule_structure == "no_structure" and sentiment_label == 1:
                            self.counters["counter_blue_area_positive"] += 1
                        elif rule_structure == "no_structure" and sentiment_label == -1:
                            self.counters["counter_blue_area_positive"] += 1

                        if rule_structure != "not_applicable" and rule_label == "no_rule":
                            self.counters["counter_no_rule_rule_syntactic_structure"] += 1

                        if (rule_label == "a-but-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_but_b_contrast"] += 1
                        if (rule_label == "a-but-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_but_b_contrast"] += 1
                        if (rule_label == "a-but-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_but_b_no_contrast"] += 1
                        if (rule_label == "a-but-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_but_b_no_contrast"] += 1

                        if (rule_label == "a-yet-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_yet_b_contrast"] += 1
                        if (rule_label == "a-yet-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_yet_b_contrast"] += 1
                        if (rule_label == "a-yet-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_yet_b_no_contrast"] += 1
                        if (rule_label == "a-yet-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_yet_b_no_contrast"] += 1

                        if (rule_label == "a-however-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_however_b_contrast"] += 1
                        if (rule_label == "a-however-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_however_b_contrast"] += 1
                        if (rule_label == "a-however-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_however_b_no_contrast"] += 1
                        if (rule_label == "a-however-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_however_b_no_contrast"] += 1

                        if (rule_label == "a-despite-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_despite_b_contrast"] += 1
                        if (rule_label == "a-despite-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_despite_b_contrast"] += 1
                        if (rule_label == "a-despite-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_despite_b_no_contrast"] += 1
                        if (rule_label == "a-despite-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_despite_b_no_contrast"] += 1

                        if (rule_label == "a-although-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_although_b_contrast"] += 1
                        if (rule_label == "a-although-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_although_b_contrast"] += 1
                        if (rule_label == "a-although-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_although_b_no_contrast"] += 1
                        if (rule_label == "a-although-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_although_b_no_contrast"] += 1

                        if (rule_label == "a-though-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_though_b_contrast"] += 1
                        if (rule_label == "a-though-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_though_b_contrast"] += 1
                        if (rule_label == "a-though-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_though_b_no_contrast"] += 1
                        if (rule_label == "a-though-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_though_b_no_contrast"] += 1

                        if (rule_label == "a-while-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_while_b_contrast"] += 1
                        if (rule_label == "a-while-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_while_b_contrast"] += 1
                        if (rule_label == "a-while-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_while_b_no_contrast"] += 1
                        if (rule_label == "a-while-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_while_b_no_contrast"] += 1

                        if (rule_label == "a-nevertheless-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_nevertheless_b_contrast"] += 1
                        if (rule_label == "a-nevertheless-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_nevertheless_b_contrast"] += 1
                        if (rule_label == "a-nevertheless-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_nevertheless_b_no_contrast"] += 1
                        if (rule_label == "a-nevertheless-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_nevertheless_b_no_contrast"] += 1

                        if (rule_label == "a-otherwise-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_otherwise_b_contrast"] += 1
                        if (rule_label == "a-otherwise-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_otherwise_b_contrast"] += 1
                        if (rule_label == "a-otherwise-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_otherwise_b_no_contrast"] += 1
                        if (rule_label == "a-otherwise-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_otherwise_b_no_contrast"] += 1
                        
                        if (rule_label == "a-still-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_still_b_contrast"] += 1
                        if (rule_label == "a-still-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_still_b_contrast"] += 1
                        if (rule_label == "a-still-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_still_b_no_contrast"] += 1
                        if (rule_label == "a-still-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_still_b_no_contrast"] += 1
                        
                        if (rule_label == "a-nonetheless-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_nonetheless_b_contrast"] += 1
                        if (rule_label == "a-nonetheless-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_nonetheless_b_contrast"] += 1
                        if (rule_label == "a-nonetheless-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_nonetheless_b_no_contrast"] += 1
                        if (rule_label == "a-nonetheless-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_nonetheless_b_no_contrast"] += 1
                        
                        if (rule_label == "a-till-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_till_b_contrast"] += 1
                        if (rule_label == "a-till-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_till_b_contrast"] += 1
                        if (rule_label == "a-till-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_till_b_no_contrast"] += 1
                        if (rule_label == "a-till-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_till_b_no_contrast"] += 1
                        
                        if (rule_label == "a-until-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_until_b_contrast"] += 1
                        if (rule_label == "a-until-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_until_b_contrast"] += 1
                        if (rule_label == "a-until-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_until_b_no_contrast"] += 1
                        if (rule_label == "a-until-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_until_b_no_contrast"] += 1
                        
                        if (rule_label == "a-in spite-b") and contrast == "contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_in spite_b_contrast"] += 1
                        if (rule_label == "a-in spite-b") and contrast == "contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_in spite_b_contrast"] += 1
                        if (rule_label == "a-in spite-b") and contrast == "no_contrast" and sentiment_label == 1:
                            self.counters["counter_positive_a_in spite_b_no_contrast"] += 1
                        if (rule_label == "a-in spite-b") and contrast == "no_contrast" and sentiment_label == -1:
                            self.counters["counter_negative_a_in spite_b_no_contrast"] += 1
                        
                        # Update the tweets in distribution counter after append
                        self.counters["tweets_in_distribution"] += 1

                        print("total number of tweets processed from the corpus: ", self.counters["tweets_in_corpus"])
                        print("total number of tweets in distribution: ", self.counters["tweets_in_distribution"])
                        print("\n")

                        # Save all the counters
                        if not os.path.exists("assets/processed_dataset"):
                            os.makedirs("assets/processed_dataset")
                        with open("assets/processed_dataset/"+self.config["asset_name"]+"_corona_tweets_"+self.config["folder_number"]+"_counters.pickle", 'wb') as handle:
                            pickle.dump(self.counters, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        # # Apply the break conditions on the counter
                        # if self.counters["counter_blue_area_positive"] == 300000 and self.counters["counter_blue_area_positive"] == 300000 and (self.counters["counter_positive_a_but_b_contrast"] == 25000 and self.counters["counter_negative_a_but_b_contrast"] == 25000 and self.counters["counter_positive_a_but_b_no_contrast"] == 25000 and self.counters["counter_negative_a_but_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_yet_b_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_contrast"] == 25000 and self.counters["counter_positive_a_yet_b_no_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_however_b_contrast"] == 25000 and self.counters["counter_negative_a_however_b_contrast"] == 25000 and self.counters["counter_positive_a_however_b_no_contrast"] == 25000 and self.counters["counter_negative_a_however_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_while_b_contrast"] == 25000 and self.counters["counter_negative_a_while_b_contrast"] == 25000 and self.counters["counter_positive_a_while_b_no_contrast"] == 25000 and self.counters["counter_negative_a_while_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_despite_b_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_contrast"] == 25000 and self.counters["counter_positive_a_despite_b_no_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_though_b_contrast"] == 25000 and self.counters["counter_negative_a_though_b_contrast"] == 25000 and self.counters["counter_positive_a_though_b_no_contrast"] == 25000 and self.counters["counter_negative_a_though_b_no_contrast"] == 25000):
                        #     break

                        pbar.update()
                    
                    except TimeoutException:
                        print("\n"+line)
                        print("\nTimeout in processing this tweet")
                        pbar.update()
                        continue
                    
                    else:
                        signal.alarm(0)

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
        print("a-but-b rule, contrast and Positive: ", self.counters["counter_positive_a_but_b_contrast"])
        print("a-but-b rule, contrast and Negative: ", self.counters["counter_negative_a_but_b_contrast"])
        print("a-but-b rule, no contrast and Positive: ", self.counters["counter_positive_a_but_b_no_contrast"])
        print("a-but-b rule, no contrast and Negative: ", self.counters["counter_negative_a_but_b_no_contrast"])
        print("\n")
        print("a-yet-b rule, contrast and Positive: ", self.counters["counter_positive_a_yet_b_contrast"])
        print("a-yet-b rule, contrast and Negative: ", self.counters["counter_negative_a_yet_b_contrast"])
        print("a-yet-b rule, no contrast and Positive: ", self.counters["counter_positive_a_yet_b_no_contrast"])
        print("a-yet-b rule, no contrast and Negative: ", self.counters["counter_negative_a_yet_b_no_contrast"])
        print("\n")
        print("a-however-b rule, contrast and Positive: ", self.counters["counter_positive_a_however_b_contrast"])
        print("a-however-b rule, contrast and Negative: ", self.counters["counter_negative_a_however_b_contrast"])
        print("a-however-b rule, no contrast and Positive: ", self.counters["counter_positive_a_however_b_no_contrast"])
        print("a-however-b rule, no contrast and Negative: ", self.counters["counter_negative_a_however_b_no_contrast"])
        print("\n")
        print("a-despite-b rule, contrast and Positive: ", self.counters["counter_positive_a_despite_b_contrast"])
        print("a-despite-b rule, contrast and Negative: ", self.counters["counter_negative_a_despite_b_contrast"])
        print("a-despite-b rule, no contrast and Positive: ", self.counters["counter_positive_a_despite_b_no_contrast"])
        print("a-despite-b rule, no contrast and Negative: ", self.counters["counter_negative_a_despite_b_no_contrast"])
        print("\n")
        print("a-although-b rule, contrast and Positive: ", self.counters["counter_positive_a_although_b_contrast"])
        print("a-although-b rule, contrast and Negative: ", self.counters["counter_negative_a_although_b_contrast"])
        print("a-although-b rule, no contrast and Positive: ", self.counters["counter_positive_a_although_b_no_contrast"])
        print("a-although-b rule, no contrast and Negative: ", self.counters["counter_negative_a_although_b_no_contrast"])
        print("\n")
        print("a-though-b rule, contrast and Positive: ", self.counters["counter_positive_a_though_b_contrast"])
        print("a-though-b rule, contrast and Negative: ", self.counters["counter_negative_a_though_b_contrast"])
        print("a-though-b rule, no contrast and Positive: ", self.counters["counter_positive_a_though_b_no_contrast"])
        print("a-though-b rule, no contrast and Negative: ", self.counters["counter_negative_a_though_b_no_contrast"])
        print("\n")
        print("a-while-b rule, contrast and Positive: ", self.counters["counter_positive_a_while_b_contrast"])
        print("a-while-b rule, contrast and Negative: ", self.counters["counter_negative_a_while_b_contrast"])
        print("a-while-b rule, no contrast and Positive: ", self.counters["counter_positive_a_while_b_no_contrast"])
        print("a-while-b rule, no contrast and Negative: ", self.counters["counter_negative_a_while_b_no_contrast"])
        print("\n")
        print("a-nevertheless-b rule, contrast and Positive: ", self.counters["counter_positive_a_nevertheless_b_contrast"])
        print("a-nevertheless-b rule, contrast and Negative: ", self.counters["counter_negative_a_nevertheless_b_contrast"])
        print("a-nevertheless-b rule, no contrast and Positive: ", self.counters["counter_positive_a_nevertheless_b_no_contrast"])
        print("a-nevertheless-b rule, no contrast and Negative: ", self.counters["counter_negative_a_nevertheless_b_no_contrast"])
        print("\n")
        print("a-otherwise-b rule, contrast and Positive: ", self.counters["counter_positive_a_otherwise_b_contrast"])
        print("a-otherwise-b rule, contrast and Negative: ", self.counters["counter_negative_a_otherwise_b_contrast"])
        print("a-otherwise-b rule, no contrast and Positive: ", self.counters["counter_positive_a_otherwise_b_no_contrast"])
        print("a-otherwise-b rule, no contrast and Negative: ", self.counters["counter_negative_a_otherwise_b_no_contrast"])
        print("\n")
        print("a-still-b rule, contrast and Positive: ", self.counters["counter_positive_a_still_b_contrast"])
        print("a-still-b rule, contrast and Negative: ", self.counters["counter_negative_a_still_b_contrast"])
        print("a-still-b rule, no contrast and Positive: ", self.counters["counter_positive_a_still_b_no_contrast"])
        print("a-still-b rule, no contrast and Negative: ", self.counters["counter_negative_a_still_b_no_contrast"])
        print("\n")
        print("a-nonetheless-b rule, contrast and Positive: ", self.counters["counter_positive_a_nonetheless_b_contrast"])
        print("a-nonetheless-b rule, contrast and Negative: ", self.counters["counter_negative_a_nonetheless_b_contrast"])
        print("a-nonetheless-b rule, no contrast and Positive: ", self.counters["counter_positive_a_nonetheless_b_no_contrast"])
        print("a-nonetheless-b rule, no contrast and Negative: ", self.counters["counter_negative_a_nonetheless_b_no_contrast"])
        print("\n")
        print("a-till-b rule, contrast and Positive: ", self.counters["counter_positive_a_till_b_contrast"])
        print("a-till-b rule, contrast and Negative: ", self.counters["counter_negative_a_till_b_contrast"])
        print("a-till-b rule, no contrast and Positive: ", self.counters["counter_positive_a_till_b_no_contrast"])
        print("a-till-b rule, no contrast and Negative: ", self.counters["counter_negative_a_till_b_no_contrast"])
        print("\n")
        print("a-until-b rule, contrast and Positive: ", self.counters["counter_positive_a_until_b_contrast"])
        print("a-until-b rule, contrast and Negative: ", self.counters["counter_negative_a_until_b_contrast"])
        print("a-until-b rule, no contrast and Positive: ", self.counters["counter_positive_a_until_b_no_contrast"])
        print("a-until-b rule, no contrast and Negative: ", self.counters["counter_negative_a_until_b_no_contrast"])
        print("\n")
        print("a-in spite-b rule, contrast and Positive: ", self.counters["counter_positive_a_in spite_b_contrast"])
        print("a-in spite-b rule, contrast and Negative: ", self.counters["counter_negative_a_in spite_b_contrast"])
        print("a-in spite-b rule, no contrast and Positive: ", self.counters["counter_positive_a_in spite_b_no_contrast"])
        print("a-in spite-b rule, no contrast and Negative: ", self.counters["counter_negative_a_in spite_b_no_contrast"])
        print("\n")

        sys.stdout.close()
        sys.stdout = old_stdout

        # Save the output data
        with open("assets/processed_dataset/"+self.config["asset_name"]+"_corona_tweets_"+self.config["folder_number"]+"_processed_data.pickle", 'wb') as handle:
            pickle.dump(self.output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save the counters
        with open("assets/processed_dataset/"+self.config["asset_name"]+"_corona_tweets_"+self.config["folder_number"]+"_counters.pickle", 'wb') as handle:
            pickle.dump(self.counters, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # # Apply the break conditions on the counter
        # if self.counters["counter_blue_area_positive"] == 300000 and self.counters["counter_blue_area_positive"] == 300000 and (self.counters["counter_positive_a_but_b_contrast"] == 25000 and self.counters["counter_negative_a_but_b_contrast"] == 25000 and self.counters["counter_positive_a_but_b_no_contrast"] == 25000 and self.counters["counter_negative_a_but_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_yet_b_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_contrast"] == 25000 and self.counters["counter_positive_a_yet_b_no_contrast"] == 25000 and self.counters["counter_negative_a_yet_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_however_b_contrast"] == 25000 and self.counters["counter_negative_a_however_b_contrast"] == 25000 and self.counters["counter_positive_a_however_b_no_contrast"] == 25000 and self.counters["counter_negative_a_however_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_while_b_contrast"] == 25000 and self.counters["counter_negative_a_while_b_contrast"] == 25000 and self.counters["counter_positive_a_while_b_no_contrast"] == 25000 and self.counters["counter_negative_a_while_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_despite_b_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_contrast"] == 25000 and self.counters["counter_positive_a_despite_b_no_contrast"] == 25000 and self.counters["counter_negative_a_despite_b_no_contrast"] == 25000) and (self.counters["counter_positive_a_though_b_contrast"] == 25000 and self.counters["counter_negative_a_though_b_contrast"] == 25000 and self.counters["counter_positive_a_though_b_no_contrast"] == 25000 and self.counters["counter_negative_a_though_b_no_contrast"] == 25000):
        #     break

                                





                        
                    
                    



            

        