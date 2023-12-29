from nltk.tokenize import TweetTokenizer

def conjunction_analysis(tweet):

    #lowercase and tokenize the sentence
    # tokenized_sentence = tweet.lower().split()
    tweet_lower_case = tweet.lower()
    tokenizer = TweetTokenizer()
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
            rule_structure = "a-but-b"
            rule_conjunct = "b"

            A_clause = tweet.split('but')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('but')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')
    
    # A-while-B
    word_list = marker_list.copy()
    word_list.remove("while")
    if ('while' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('while') != 0 and tokenized_sentence.index('while') != -1 and tokenized_sentence.count('while') == 1):
            rule_structure = "a-while-b"
            rule_conjunct = "a"

            A_clause = tweet.split('while')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('while')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-yet-B
    word_list = marker_list.copy()
    word_list.remove("yet")
    if ('yet' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('yet') != 0 and tokenized_sentence.index('yet') != -1 and tokenized_sentence.count('yet') == 1): # Check if the sentence contains A-yet-B structure
            rule_structure = "a-yet-b"
            rule_conjunct = "b"   

            A_clause = tweet.split('yet')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('yet')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-however-B
    word_list = marker_list.copy()
    word_list.remove("however")
    if ('however' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('however') != 0 and tokenized_sentence.index('however') != -1 and tokenized_sentence.count('however') == 1):
            rule_structure = "a-however-b"
            rule_conjunct = "b"

            A_clause = tweet.split('however')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('however')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-despite-B
    word_list = marker_list.copy()
    word_list.remove("despite")
    if ('despite' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('despite') != 0 and tokenized_sentence.index('despite') != -1 and tokenized_sentence.count('despite') == 1):
            rule_structure = "a-despite-b"
            rule_conjunct = "b"

            A_clause = tweet.split('despite')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('despite')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-although-B
    word_list = marker_list.copy()
    word_list.remove("although")
    if ('although' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('although') != 0 and tokenized_sentence.index('although') != -1 and tokenized_sentence.count('although') == 1):
            rule_structure = "a-although-b"
            rule_conjunct = "a"

            A_clause = tweet.split('although')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('although')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-though-B
    word_list = marker_list.copy()
    word_list.remove("though")
    if ('though' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('though') != 0 and tokenized_sentence.index('though') != -1 and tokenized_sentence.count('though') == 1):
            rule_structure = "a-though-b"
            rule_conjunct = "a"

            A_clause = tweet.split('though')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('though')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-nevertheless-B
    word_list = marker_list.copy()
    word_list.remove("nevertheless")
    if ('nevertheless' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('nevertheless') != 0 and tokenized_sentence.index('nevertheless') != -1 and tokenized_sentence.count('nevertheless') == 1):
            rule_structure = "a-nevertheless-b"
            rule_conjunct = "b"

            A_clause = tweet.split('nevertheless')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('nevertheless')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-otherwise-B
    word_list = marker_list.copy()
    word_list.remove("otherwise")
    if ('otherwise' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('otherwise') != 0 and tokenized_sentence.index('otherwise') != -1 and tokenized_sentence.count('otherwise') == 1):
            rule_structure = "a-otherwise-b"
            rule_conjunct = "b"

            A_clause = tweet.split('otherwise')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('otherwise')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-still-B
    word_list = marker_list.copy()
    word_list.remove("still")
    if ('still' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('still') != 0 and tokenized_sentence.index('still') != -1 and tokenized_sentence.count('still') == 1):
            rule_structure = "a-still-b"
            rule_conjunct = "b"

            A_clause = tweet.split('still')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('still')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-nonetheless-B
    word_list = marker_list.copy()
    word_list.remove("nonetheless")
    if ('nonetheless' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('nonetheless') != 0 and tokenized_sentence.index('nonetheless') != -1 and tokenized_sentence.count('nonetheless') == 1):
            rule_structure = "a-nonetheless-b"
            rule_conjunct = "b"

            A_clause = tweet.split('nonetheless')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('nonetheless')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-till-B
    word_list = marker_list.copy()
    word_list.remove("till")
    if ('till' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('till') != 0 and tokenized_sentence.index('till') != -1 and tokenized_sentence.count('till') == 1):
            rule_structure = "a-till-a"
            rule_conjunct = "a"

            A_clause = tweet.split('till')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('till')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')
    
    # A-until-B
    word_list = marker_list.copy()
    word_list.remove("until")
    if ('until' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('until') != 0 and tokenized_sentence.index('until') != -1 and tokenized_sentence.count('until') == 1):
            rule_structure = "a-until-b"
            rule_conjunct = "a"

            A_clause = tweet.split('until')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('until')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    # A-in spite-B
    word_list = marker_list.copy()
    word_list.remove("in spite")
    if ('in spite' in tokenized_sentence and not any(word in tokenized_sentence for word in word_list)):
        if (tokenized_sentence.index('in spite') != 0 and tokenized_sentence.index('in spite') != -1 and tokenized_sentence.count('in spite') == 1):
            rule_structure = "a-in spite-b"
            rule_conjunct = "a"

            A_clause = tweet.split('in spite')[0]
            A_clause = ''.join(A_clause)
            A_clause = A_clause.strip().replace('  ', ' ')

            B_clause = tweet.split('in spite')[1:]
            B_clause = ''.join(B_clause)
            B_clause = B_clause.strip().replace('  ', ' ')

    return rule_structure, rule_conjunct, A_clause, B_clause