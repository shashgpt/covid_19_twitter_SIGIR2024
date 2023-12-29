def append_positive_blue_area(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if rule_structure == "no_structure" and sentiment_label == 1: # Blue area positive tweets
        if counters["counter_blue_area_positive"] < 300000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_blue_area_positive"] += 1
    
    return counters, output_data
    
def append_negative_blue_area(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if rule_structure == "no_structure" and sentiment_label == -1: # Blue area negative tweets     
        if counters["counter_blue_area_negative"] < 300000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_blue_area_negative"] += 1
    
    return counters, output_data

def append_no_rule_rule_syntactic_structure(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if rule_structure != "not_applicable" and rule_label == "no_rule": # No rule tweets but having syntactic structure
    
        output_data['tweet_id'].append(tweet_id)
        output_data['tweet'].append(tweet)
        output_data['preprocessed_tweet'].append(preprocessed_tweet)
        output_data['clause_A'].append(clause_A)
        output_data['clause_B'].append(clause_B)
        output_data['emojis'].append(emojis)
        output_data['emoji_names'].append(emoji_names)
        output_data['emotion_scores'].append(emotion_scores)
        output_data['agg_emotion_score'].append(agg_emotion_score)
        output_data['sentiment_label'].append(sentiment_label)
        output_data['vader_score_sentence'].append(vader_score_sentence)
        output_data['vader_score_clause_A'].append(vader_score_clause_A)
        output_data['vader_score_clause_B'].append(vader_score_clause_B)
        output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
        output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
        output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
        output_data['rule_structure'].append(rule_structure)
        output_data['rule_label'].append(rule_label)
        output_data['contrast'].append(contrast)

        counters["counter_no_rule_rule_syntactic_structure"] += 1
    
    return counters, output_data

def append_positive_a_but_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "contrast" and sentiment_label == 1: # A-but-B rule
        if counters["counter_positive_a_but_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_but_b_contrast"] += 1

    return counters, output_data

def append_negative_a_but_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "contrast" and sentiment_label == -1: # A-but-B rule
        if counters["counter_negative_a_but_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_but_b_contrast"] += 1

    return counters, output_data

def append_positive_a_but_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_but_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_but_b_no_contrast"] += 1
    
    return counters, output_data

def append_negative_a_but_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-but-B" or rule_label == "A-But-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_but_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_but_b_no_contrast"] += 1

    return counters, output_data

def append_positive_a_yet_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_yet_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_yet_b_contrast"] += 1

    return counters, output_data

def append_negative_a_yet_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_yet_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_yet_b_contrast"] += 1

    return counters, output_data

def append_positive_a_yet_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_yet_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_yet_b_no_contrast"] += 1

    return counters, output_data

def append_negative_a_yet_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-yet-B" or rule_label == "A-Yet-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_yet_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_yet_b_no_contrast"] += 1

    return counters, output_data

def append_positive_a_however_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_however_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_however_b_contrast"] += 1
    
    return counters, output_data

def append_negative_a_however_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_however_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_however_b_contrast"] += 1

    return counters, output_data

def append_positive_a_however_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_however_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_however_b_no_contrast"] += 1

    return counters, output_data

def append_negative_a_however_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-however-B" or rule_label == "A-However-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_however_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_however_b_no_contrast"] += 1
    
    return counters, output_data

def append_positive_a_despite_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_despite_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_despite_b_contrast"] += 1

    return counters, output_data

def append_negative_a_despite_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_despite_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_despite_b_contrast"] += 1

    return counters, output_data

def append_positive_a_despite_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_despite_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_despite_b_no_contrast"] += 1
    
    return counters, output_data

def append_negative_a_despite_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-despite-B" or rule_label == "A-Despite-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_despite_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_despite_b_no_contrast"] += 1
    
    return counters, output_data

def append_positive_a_although_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_although_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_although_b_contrast"] += 1
    
    return counters, output_data

def append_negative_a_although_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_although_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_although_b_contrast"] += 1
    
    return counters, output_data

def append_positive_a_although_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_although_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_although_b_no_contrast"] += 1
    
    return counters, output_data

def append_negative_a_although_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-although-B" or rule_label == "A-Although-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_although_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_although_b_no_contrast"] += 1
    
    return counters, output_data

def append_positive_a_though_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_though_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_though_b_contrast"] += 1
    
    return counters, output_data

def append_negative_a_though_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_though_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_though_b_contrast"] += 1
    
    return counters, output_data

def append_positive_a_though_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_though_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_though_b_no_contrast"] += 1
    
    return counters, output_data

def append_negative_a_though_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-though-B" or rule_label == "A-Though-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_though_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_though_b_no_contrast"] += 1
    
    return counters, output_data

def append_positive_a_while_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "contrast" and sentiment_label == 1:
        if counters["counter_positive_a_while_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_while_b_contrast"] += 1
    
    return counters, output_data

def append_negative_a_while_b_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):
    
    if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "contrast" and sentiment_label == -1:
        if counters["counter_negative_a_while_b_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_while_b_contrast"] += 1
    
    return counters, output_data

def append_positive_a_while_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "no_contrast" and sentiment_label == 1:
        if counters["counter_positive_a_while_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_positive_a_while_b_no_contrast"] += 1
    
    return counters, output_data

def append_negative_a_while_b_no_contrast(counters, output_data, tweet_id, tweet, preprocessed_tweet, clause_A, clause_B, emojis, emoji_names, emotion_scores, agg_emotion_score, sentiment_label, vader_score_sentence, vader_score_clause_A, vader_score_clause_B, vader_sentiment_sentence, vader_sentiment_clause_A, vader_sentiment_clause_B, rule_structure, rule_label, contrast):

    if (rule_label == "A-while-B" or rule_label == "A-While-B") and contrast == "no_contrast" and sentiment_label == -1:
        if counters["counter_negative_a_while_b_no_contrast"] < 25000:
            output_data['tweet_id'].append(tweet_id)
            output_data['tweet'].append(tweet)
            output_data['preprocessed_tweet'].append(preprocessed_tweet)
            output_data['clause_A'].append(clause_A)
            output_data['clause_B'].append(clause_B)
            output_data['emojis'].append(emojis)
            output_data['emoji_names'].append(emoji_names)
            output_data['emotion_scores'].append(emotion_scores)
            output_data['agg_emotion_score'].append(agg_emotion_score)
            output_data['sentiment_label'].append(sentiment_label)
            output_data['vader_score_sentence'].append(vader_score_sentence)
            output_data['vader_score_clause_A'].append(vader_score_clause_A)
            output_data['vader_score_clause_B'].append(vader_score_clause_B)
            output_data['vader_sentiment_sentence'].append(vader_sentiment_sentence)
            output_data['vader_sentiment_clause_A'].append(vader_sentiment_clause_A)
            output_data['vader_sentiment_clause_B'].append(vader_sentiment_clause_B)
            output_data['rule_structure'].append(rule_structure)
            output_data['rule_label'].append(rule_label)
            output_data['contrast'].append(contrast)

            counters["counter_negative_a_while_b_no_contrast"] += 1
    
    return counters, output_data