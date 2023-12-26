def emoji_thresholds(self): # Code to calculate the Sentiment and VADER thresholds for the tweets

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