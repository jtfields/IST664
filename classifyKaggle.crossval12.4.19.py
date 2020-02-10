'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>

  This version uses cross-validation with the Naive Bayes classifier in NLTK.
  It computes the evaluation measures of precision, recall and F1 measure for each fold.
  It also averages across folds and across labels.
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
import re
import save_features
import run_sklearn_model_performance
import sentiment_read_subjectivity
import pandas
import sentiment_read_LIWC_pos_neg_words #added 12/1
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer #added 11/26
from nltk.corpus import stopwords #added 11/26
from nltk.tokenize import WordPunctTokenizer #added 11/27
from nltk.collocations import * #added 11/28

## FEATURE DEFINITION FUNCTIONS

## BAG OF WORD / UNIGRAM BASELINE FEATURES
# this function define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

## BI-GRAM FEATURES (added 11/28)
# define features that include words as before 
# add the most frequent significant bigrams
# this function takes the list of words in a document as an argument and returns a feature dictionary
# it depends on the variables word_features and bigram_features
def bigram_document_features(document, word_features, bigram_features):
    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)    
    return features

## PART OF SPEECH (POS) FEATURES (added 11/28)
# this function takes a document list of words and returns a feature dictionary
# it runs the default pos tagger (the Stanford tagger) on the document
#   and counts 4 types of pos tags to use as features
def POS_features(document, word_features):
    document_words = set(document)
    tagged_words = nltk.pos_tag(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    numNoun = 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words:
        if tag.startswith('N'): numNoun += 1
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['nouns'] = numNoun
    features['verbs'] = numVerb
    features['adjectives'] = numAdj
    features['adverbs'] = numAdverb
    return features

## SENTIMENT LEXICON: SUBJECTIVITY COUNT
#initialize the positive, neutral and negative word lists
SLpath = "/Users/johnfields/Desktop/kagglemoviereviews/SentimentLexicons/subjclueslen1-HLTEMNLP05.tff"

def readSubjectivity(path): #added 11/30
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict

# define features that include word counts of subjectivity words
# negative feature will have number of weakly negative words +
#    2 * number of strongly negative words
# positive feature has similar definition
#    not counting neutral words
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features

## SENTIMENT LEXICON: LIWC (added 12/1)

    def liwc_features(doc, word_features,poslist,neglist):
      doc_words = set(doc) 
      features = {}
      for word in word_features:
        features['contains({})'.format(word)] = (word in doc_words) 
      pos = 0
      neg = 0
      for word in doc_words:
        if sentiment_read_LIWC_pos_neg_words.isPresent(word,poslist): 
          pos += 1
        if sentiment_read_LIWC_pos_neg_words.isPresent(word,neglist): 
          neg += 1
        features['positivecount'] = pos
        features['negativecount'] = neg 
      if 'positivecount' not in features:
        features['positivecount']=0
      if 'negativecount' not in features:
        features['negativecount']=0 
      return features

## NEGATION WORDS
# this list of negation words includes some "approximate negators" like hardly and rarely
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

# One strategy with negation words is to negate the word following the negation word
#   other strategies negate all words up to the next punctuation
# Strategy is to go through the document words in order adding the word features,
#   but if the word follows a negation words, change the feature to negated word
# Start the feature set with all 2000 word features and 2000 Not word features set to false
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features


## CROSS-VALIDATION
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        print('Fold', i)
        (precision_list, recall_list, F1_list) \
                  = eval_measures(goldlist, predictedlist, labels)
        # take off triple string to print precision, recall and F1 for each fold
        '''
        print('\tPrecision\tRecall\t\tF1')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        '''
        # for each label add to the sums in the total lists
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels), \
          "{:10.3f}".format(sum(recall_list)/num_labels), \
          "{:10.3f}".format(sum(F1_list)/num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision), \
      "{:10.3f}".format(recall), "{:10.3f}".format(F1))
    

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output: returns lists of precision, recall and F1 for each label
#      (for computing averages across folds and labels)
def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)

##LOAD DATA
# function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
 
##PRE-PROCESSING (Added 11/26)
  
  stopwords = nltk.corpus.stopwords.words('english')
  newstopwords = [word for word in stopwords if word not in ['not','no','can','don','t']]  

  def pre_processing_documents(document):
    # "Pre_processing_documents"  
    # "create list of lower case words"
    word_list = re.split('\s+', document.lower())
    # punctuation and numbers to be removed
    punctuation = re.compile(r'[-.?!/\%@,":;()|0-9]')
    word_list = [punctuation.sub("", word) for word in word_list] 
    final_word_list = []
    for word in word_list:
      if word not in newstopwords:
        final_word_list.append(word)
    line = " ".join(final_word_list)
    return line

##RANDOMIZE 
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  phrasedocs_without = [] #added 11/26
  # add all the phrases

##TOKENIZE
  
  # NLK standard
  #each phrase has a list of tokens and the sentiment label (from 0 to 4)
  ### bin to only 3 categories for better performance
  for phrase in phraselist:
    #without pre-processing (changed 11/26)
    #tokens = nltk.word_tokenize(phrase[0])
    #phrasedocs.append((tokens, int(phrase[1])))

  # Regexp
    #with pre-processing (added 11/26)
    #tokenizer = RegexpTokenizer(r'\w+')
    #phrase[0] = pre_processing_documents(phrase[0])
    #tokens = tokenizer.tokenize(phrase[0])
    #phrasedocs.append((tokens,int(phrase[1])))

  # Word_punct
    tokenizer = WordPunctTokenizer()
    phrase[0] = pre_processing_documents(phrase[0])
    tokens = tokenizer.tokenize(phrase[0])
    phrasedocs.append((tokens,int(phrase[1])))    


##FILTER TOKENS
  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  docs = []
  for phrase in phrasedocs:
    lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
    docs.append (lowerphrase)
  # print a few
  for phrase in docs[:10]:
    print (phrase)

## CREATE WORD FEATURES

# continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)
  print(len(all_words))

  # get the 1500 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(1500)
  word_features = [word for (word,count) in word_items]

##SUBJECTIVITY (added 11/30)
  SL = readSubjectivity(SLpath)
  featuresets = [(SL_features(d, word_features, SL), c) for (d, c) in docs]

##LIWC SENTIMENT (added 12/1)
  #poslist,neglist = sentiment_read_LIWC_pos_neg_words.read_words() 
  #featuresets = [(document_features(d, word_features), c) for (d, c) in docs]

##UNIGRAMS
  #feature sets from a feature definition function
  #featuresets = [(document_features(d, word_features), c) for (d, c) in docs]

##BIGRAMS (added 11/28)
  
  #bigram_measures = nltk.collocations.BigramAssocMeasures()

  # create the bigram finder on all the words in sequence
  #finder = BigramCollocationFinder.from_words(all_words_list)

  # define the top 500 bigrams using the chi squared measure
  # can also use raw count or PMI
  #bigram_features = finder.nbest(bigram_measures.chi_sq, 500)
  #print(bigram_features[:50])

  # use this function to create feature sets for all sentences
  #bigram_featuresets = [(bigram_document_features(d, word_features, bigram_features), c) for (d, c) in docs]

## POS (added 11/28)

  # define feature sets using this function
  #POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in docs]
  #number of features for document 0
  #print(len(POS_featuresets[0][0].keys()))

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN CROSS-VALIDATION - UNIGRAMS
  # make a list of labels
  #label_list = [c for (d,c) in docs]
  #labels = list(set(label_list))    # gets only unique labels
  #num_folds = 5
  #cross_validation_PRF(num_folds, featuresets, labels)

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN CROSS-VALIDATION - BIGRAMS #Added 11/28
  #make a list of labels
  #label_list = [c for (d,c) in docs]
  #labels = list(set(label_list))    # gets only unique labels
  #num_folds = 5
  #cross_validation_PRF(num_folds, bigram_featuresets, labels)

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN CROSS-VALIDATION - SL FEATURES #Added 11/30
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  cross_validation_PRF(num_folds, featuresets, labels)

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN CROSS-VALIDATION - LIWC FEATURES #Added 12/1
  #make a list of labels
  #label_list = [c for (d,c) in docs]
  #labels = list(set(label_list))    # gets only unique labels
  #num_folds = 5
  #cross_validation_PRF(num_folds, featuresets, labels)

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN CROSS-VALIDATION - POS #Added 11/28
  #make a list of labels
  #label_list = [c for (d,c) in docs]
  #labels = list(set(label_list))    # gets only unique labels
  #num_folds = 5
  #cross_validation_PRF(num_folds, POS_featuresets, labels)

## TRAIN CLASSIFIER AND SHOW PERFORMANCE IN NEGATION WORDS #Added 11/30
  # define the feature sets
  NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in docs]
  # make a list of labels
  #label_list = [c for (d,c) in docs]
  #labels = list(set(label_list))    # gets only unique labels
  #num_folds = 5
  #cross_validation_PRF(num_folds, NOT_featuresets, labels)

## SAVE FEATURES TO CSV FILE
  save_features.writeFeatureSets(featuresets,"features.csv") #Added 11/28

## SKLEARN EXTERNAL CLASSIFIER
  #run_sklearn_model_performance.process /Users/johnfields/Desktop/kagglemoviereviews/corpus/features.csv


"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])