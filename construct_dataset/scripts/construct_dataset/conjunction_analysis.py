from nltk.tokenize import TweetTokenizer

def conjunction_analysis(tweet):

    #lowercase and tokenize the sentence
    # tokenized_sentence = preprocessed_tweet.lower().split()
    tweet_lower_case = tweet.lower()
    tokenizer = TweetTokenizer
    tokenized_sentence = tokenizer.tokenize(tweet_lower_case)

    #output
    rule_structure = None
    rule_conjunct = None
    A_clause = None
    B_clause = None

    #markers list
    marker_list = [
                    "but",
                    "yet",
                    "though",
                    "while",
                    "however",
                    "despite",
                    "though",
                    "although",
                    "nevertheless",
                    "otherwise",
                    "still",
                    "nonetheless",
                    "till",
                    "until",
                    "in spite"
                  ]

    # A-but-B
    word_list = marker_list.copy()
    word_list.remove("but")
    if ('but' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('but') != 0 and tokenized_sentence.index('but') != -1 and tokenized_sentence.count('but') == 1): # Check if the sentence contains A-but-B structure
            rule_structure = "A-but-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("but")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("but")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')
    
    # A-while-B
    word_list = marker_list.copy()
    word_list.remove("while")
    if ('while' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
            rule_structure = "A-while-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("while")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("while")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-yet-B
    word_list = marker_list.copy()
    word_list.remove("yet")
    if ('yet' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1): # Check if the sentence contains A-yet-B structure
            rule_structure = "A-yet-B"
            rule_conjunct = "B"   

            A_clause = tokenized_sentence[:tokenized_sentence.index("yet")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("yet")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-however-B
    word_list = marker_list.copy()
    word_list.remove("however")
    if ('however' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('however') != 0 and tokenized_sentence.index('however') != -1 and tokenized_sentence.count('however') == 1):
            rule_structure = "A-however-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("however")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("however")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-despite-B
    word_list = marker_list.copy()
    word_list.remove("despite")
    if ('despite' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('despite') != 0 and tokenized_sentence.index('despite') != -1 and tokenized_sentence.count('despite') == 1):
            rule_structure = "A-despite-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("despite")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("despite")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-although-B
    word_list = marker_list.copy()
    word_list.remove("although")
    if ('although' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('although') != 0 and tokenized_sentence.index('although') != -1 and tokenized_sentence.count('although') == 1):
            rule_structure = "A-although-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("although")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("although")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-though-B
    word_list = marker_list.copy()
    word_list.remove("though")
    if ('though' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
            rule_structure = "A-though-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("though")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("though")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-nevertheless-B
    word_list = marker_list.copy()
    word_list.remove("nevertheless")
    if ('nevertheless' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('nevertheless') != 0 and tokenized_sentence.index('nevertheless') != -1 and tokenized_sentence.count('nevertheless') == 1):
            rule_structure = "A-nevertheless-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("nevertheless")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("nevertheless")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-otherwise-B
    word_list = marker_list.copy()
    word_list.remove("otherwise")
    if ('otherwise' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('otherwise') != 0 and tokenized_sentence.index('otherwise') != -1 and tokenized_sentence.count('otherwise') == 1):
            rule_structure = "A-otherwise-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("otherwise")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("otherwise")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-still-B
    word_list = marker_list.copy()
    word_list.remove("still")
    if ('still' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('still') != 0 and tokenized_sentence.index('still') != -1 and tokenized_sentence.count('still') == 1):
            rule_structure = "A-still-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("still")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("still")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-nonetheless-B
    word_list = marker_list.copy()
    word_list.remove("nonetheless")
    if ('nonetheless' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('nonetheless') != 0 and tokenized_sentence.index('nonetheless') != -1 and tokenized_sentence.count('nonetheless') == 1):
            rule_structure = "A-nonetheless-B"
            rule_conjunct = "B"

            A_clause = tokenized_sentence[:tokenized_sentence.index("nonetheless")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("nonetheless")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-till-B
    word_list = marker_list.copy()
    word_list.remove("till")
    if ('till' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('till') != 0 and tokenized_sentence.index('till') != -1 and tokenized_sentence.count('till') == 1):
            rule_structure = "A-till-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("till")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("till")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')
    
    # A-until-B
    word_list = marker_list.copy()
    word_list.remove("until")
    if ('until' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('until') != 0 and tokenized_sentence.index('until') != -1 and tokenized_sentence.count('until') == 1):
            rule_structure = "A-until-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("until")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("until")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-in spite-B
    word_list = marker_list.copy()
    word_list.remove("in spite")
    if ('in spite' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('in spite') != 0 and tokenized_sentence.index('in spite') != -1 and tokenized_sentence.count('in spite') == 1):
            rule_structure = "A-in spite-B"
            rule_conjunct = "A"

            A_clause = tokenized_sentence[:tokenized_sentence.index("in spite")]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tokenized_sentence[tokenized_sentence.index("in spite")+1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    if (A_clause ==  None and B_clause == None and rule_structure == None and rule_conjunct == None):
        return None, None, None, None