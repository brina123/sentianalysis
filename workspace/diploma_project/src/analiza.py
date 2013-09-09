# -*- coding: utf-8 -*-

"""
analiza.py 
"""

import sys
import os
import time
import re
import csv
import MySQLdb
import xml.etree.cElementTree as et

import nltk 

from sklearn import metrics
from sklearn.cross_validation import ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pylab as pl
import numpy as np

# ===========================================================
# print log 
# ===========================================================

#import logging
logs = []
def print_log(msg):
    #logging.info(msg)
    global logs
    logs.append(msg)
    print msg

def check_create_dir(filepathname):
    (filepath, filename) = os.path.split(filepathname) 
    filepathdir = os.path.dirname(filepath + "\\")
    if not os.path.exists(filepathdir):
        os.makedirs(filepathdir)

def write_log(logfilename):
    check_create_dir(logfilename)
    logfile = open(logfilename, "wb+")
    for msg in logs:
        logfile.write(msg + "\n")
    logfile.close()

# ===========================================================
# random razdelitev 
# ===========================================================

def create_csvtrain_and_csvtest(random_csvfilename, train_csvfilename, test_csvfilename, random_col, meja): 
    csvfile_random = open(random_csvfilename, "rb")
    csvfile_train = open(train_csvfilename, "wb+")
    csvfile_test  = open(test_csvfilename, "wb+")
    rows = csv.reader(csvfile_random, delimiter=';', quotechar='"')
    csvwriter_train = csv.writer(csvfile_train, delimiter=';', quotechar='"')
    csvwriter_test = csv.writer(csvfile_test, delimiter=';', quotechar='"')
    for cols in rows:
        id       = cols[0]
        classify = cols[1]
        if int(cols[random_col]) > meja:
            csvwriter_test.writerow((id, classify))
        else:
            #print_log(cols[random_col])
            csvwriter_train.writerow((id, classify))
    csvfile_train.close()
    csvfile_test.close()
    csvfile_random.close()    


# ===========================================================
# corpus 
# ===========================================================

def load_corpus_form_xmlfile(xmlfilename, namespace):
    corpus = {}
    tree = et.ElementTree(file=xmlfilename)
    for elem in tree.iter(tag="{%s}w" % namespace):
        word  = elem.text.lower()
        lemma = elem.get("lemma").lower()
        if (word not in corpus) or corpus[word] == lemma:
            corpus[word] = lemma    
    return corpus

def save_corpus_to_csvfile(corpus, corpus_csvfilename):
    csvfile = open(corpus_csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"')
    for word in corpus:
        csvwriter.writerow((word, corpus[word]))
    csvfile.close()

def load_corpus_form_csvfile(csvfilename):
    corpus = {}
    csvfile = open(csvfilename, "rb")
    rows = csv.reader(csvfile, delimiter=';', quotechar='"')
    for word, lemma in rows:
        corpus[word] = lemma    
    csvfile.close()
    return corpus

def get_corpus(corpus_name, corpus_namespace):
    corpus_xml = corpus_name + ".xml"
    corpus_csv = corpus_name + ".csv"
                
    if not os.path.exists(corpus_csv):
        # prvič prepiše v csv
        corpus = load_corpus_form_xmlfile(corpus_xml, corpus_namespace)
        save_corpus_to_csvfile(corpus, corpus_csv)
    else:
        # nato uporabi csv
        corpus = load_corpus_form_csvfile(corpus_csv)        
        
    print_log("  corpus words = %i" % len(corpus))
    
    return corpus

# ===========================================================
# stop words 
# ===========================================================

def load_words_from_txtfile(txtfilename):
    txtfile = open(txtfilename)
    words_text = txtfile.read()
    txtfile.close()
    words = words_text.lower().split()
    return words

def get_stopwords(stopwords_name, corpus):
    #stop_words = nltk.corpus.stop_words.words("english")
    stop_words = load_words_from_txtfile(stopwords_name + ".txt")
    if corpus != None:
        (stop_words, replace) = get_corpus_replace_words(stop_words, corpus)
        #print "  stop_words replace = %i/%i " % (replace, len(stop_words)) 
    print_log("  stop words = %i " % len(stop_words))    
    return stop_words    

# ===========================================================
# load
# ===========================================================

def clean_min_max_words(words, min_word_length, max_word_length):
    clean = 0        
    cwords = []
    for word in words:
        if min_word_length == None or max_word_length == None or min_word_length <= len(word) <= max_word_length:
            cwords.append(word)
        else:        
            clean += 1
    return (cwords, clean)

def clean_stop_words(words, stop_words):
    clean = 0        
    cwords = []
    for word in words:
        if stop_words == None or not word in stop_words:
            cwords.append(word)
        else:
            clean += 1        
    return (cwords, clean)

def clean_doublechars(beseda):
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
    return pattern.sub(r"\1\1", beseda)


def get_corpus_replace_words(words, corpus):
    cwords = []
    replace = 0
    for word in words:
        if word in corpus:
            cwords.append(corpus[word])
            replace += 1
        else:      
            cwords.append(word)
    return (cwords, replace)

def get_neighbors_words(words):
    neighbor_words = []
    pword = None
    for word in words:
        if pword != None:
            twowords = pword + '/' + word
            neighbor_words.append(twowords)
        pword = word
    return neighbor_words 

def get_clean_words(words, corpus, min_word_length, max_word_length, stop_words, use_neighbors):
    (words, clean) = clean_min_max_words(words, min_word_length, max_word_length)
        
    if corpus != None:
        (words, replace) = get_corpus_replace_words(words, corpus)
    else: 
        replace = 0
            
    (words, sw_clean) = clean_stop_words(words, stop_words)        
    clean += sw_clean            

    if use_neighbors:    
        words = get_neighbors_words(words)
    
    return (words, clean, replace)
        
def get_words_from_besedilo(besedilo):
    besedilo = re.sub("\d+", "", besedilo)
    besedilo = clean_doublechars(besedilo)
    wordstext = besedilo.replace(',', ' ').replace('.', ' ').replace('*', ' ').replace('!', ' ').replace('?', ' ')    #  ':)'
    words = wordstext.lower().split()
    return words

def get_words_from_db(db, id):
    cursor = db.cursor()
    cursor.execute("SELECT besedilo FROM komentar WHERE idKomentar = %s" % (id))
    besedilo = cursor.fetchone()[0]
    words = get_words_from_besedilo(besedilo)
    return words

def load_data_from_csv(msg, db, csvfilename, corpus, min_word_length, max_word_length, stop_words, use_neighbors): 
    t0 = time.time()
    print_log("\nload " + msg)

    csvfile = open(csvfilename, "rb")
    rows = csv.reader(csvfile, delimiter=';', quotechar='"')
    data = []
    words_count = 0
    clean_count = 0
    replace_count = 0
    
    for id, classify in rows:
        words = get_words_from_db(db, id)
        words_count += len(words) 
        (words, clean, replace) = get_clean_words(words, corpus, min_word_length, max_word_length, stop_words, use_neighbors)
        if len(words) > 0: 
            clean_count += clean
            replace_count += replace
            data.append((id, classify, words))
        
    csvfile.close()
    
    if corpus != None:
        if words_count > 0:
            replace_procent = replace_count * 100 / words_count    
        else:
            replace_procent = 0                    
        print_log("  corpus = %i%% %i/%i" % (replace_procent, replace_count, words_count))
    if clean_count > 0:            
        print_log("  clean(stop,min,max) = %i" % clean_count)                

    print_data_classifiers(data)    
    print_log("  čas = %i ms" % ((time.time() - t0) * 1000))
    return data

# ===========================================================
# save
# ===========================================================

def get_data_from_db(db, id):
    cursor = db.cursor()
    cursor.execute("SELECT plusi, minusi, besedilo FROM komentar WHERE idKomentar = %s" % (id))
    (plusi, minusi, besedilo) = cursor.fetchone()
    return (plusi, minusi, besedilo)

def save_data_notok_to_csv(db, test_data, test_new_classify_set, csvfilename, none_classify):
    print_log("\nsave")
    t0 = time.time()
    check_create_dir(csvfilename)
    csvfile = open(csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"')
    csvwriter.writerow(("id", "classify", "new_classify", "not_ok", "plusi", "minusi", "words", "besedilo"))
    row = 0
    for id, classify, words in test_data:
        new_classify = test_new_classify_set[row]        
        not_ok = '#' if new_classify != none_classify and new_classify != classify else None
        (plusi, minusi, besedilo) = get_data_from_db(db, id)
        wirdstext = ''
        for word in words:
            wirdstext += word + ','
        # ------------    
        csvwriter.writerow((id, classify, new_classify, not_ok, plusi, minusi, wirdstext, besedilo))
        # ------------    
        row += 1
    csvfile.close()
    print_log("  save %s = %i ms" % (csvfilename, (time.time() - t0) * 1000))

def save_data_to_csv(db, test_data, test_new_classify_set, csvfilename):
    t0 = time.time()
    check_create_dir(csvfilename)
    csvfile = open(csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"')
    csvwriter.writerow(("id", "classify", "plusi", "minusi", "besedilo"))
    row = 0
    for id, classify, words in test_data:
        new_classify = test_new_classify_set[row]
        (plusi, minusi, besedilo) = get_data_from_db(db, id)
        # ------------    
        csvwriter.writerow((id, new_classify, plusi, minusi, besedilo))
        # ------------    
        row += 1
    csvfile.close()
    print_log("  save %s = %i ms" % (csvfilename, (time.time() - t0) * 1000))

def save_new_classify_to_csv(db, test_data, test_new_classify_set, csvfilename):
    t0 = time.time()
    check_create_dir(csvfilename)
    csvfile = open(csvfilename, "wb+")
    csvwriter = csv.writer(csvfile, delimiter=';', quotechar='"')
    row = 0
    for id, classify, words in test_data:
        new_classify = test_new_classify_set[row]
        # ------------    
        csvwriter.writerow((id, new_classify))
        # ------------    
        row += 1
    csvfile.close()
    print_log("  save %s  čas = %i ms" % (csvfilename, (time.time() - t0) * 1000))

# ===========================================================
# data
# ===========================================================

def get_index_data(s_index, data):
    s_data = []
    for index in s_index:
        s_data.append(data[index])
    #print_data_classifiers(s_data)
    return s_data

def get_classify_set(data):
    classify_set = [] 
    for id, classify, words in data:
        classify_set.append(classify)
    return classify_set

def get_words_set(data):
    words_set = [] 
    for id, classify, words in data:
        text = None
        for word in words:
            if text == None:
                text = ''
            else:
                text += ' ' 
            text += word 
        words_set.append(text)
    return words_set

n_count = 0
n_words = 1
def print_classifiers(classifiers):
    msg = "  "
    sum_classifiers = {}
    sum_classifiers[n_count] = 0
    sum_classifiers[1] = 0
    for classify in classifiers:
        msg += "(%s)=%i/%i, " % (classify, classifiers[classify][n_count], classifiers[classify][n_words])
        sum_classifiers[n_count] += classifiers[classify][n_count]
        sum_classifiers[n_words] += classifiers[classify][n_words]
    msg += "vsi=%i/%i" % (sum_classifiers[n_count], sum_classifiers[n_words])
    print_log(msg)
        
def print_data_classifiers(data):
    classifiers = {}            
    for id, classify, words in data:
        if classify not in classifiers:
            classifiers[classify] = {}      
            classifiers[classify][n_count] = 0       
            classifiers[classify][n_words] = 0       
        classifiers[classify][n_count] += 1
        classifiers[classify][n_words] += len(words)
    print_classifiers(classifiers)    

def print_new_classify_set(classify_set):
    classifiers = {}            
    for classify in classify_set:
        if classify not in classifiers:
            classifiers[classify] = {}      
            classifiers[classify][n_count] = 0       
            classifiers[classify][n_words] = 0       
        classifiers[classify][n_count] += 1
        #classifiers[classify][n_words] += len(words)
    print_classifiers(classifiers)    

# ===========================================================
# nltk    
# ===========================================================

def nltk_get_wordsclassify_set(data): 
    wordsclassify_set = [] 
    for id, classify, words in data:
        wordsclassify_set.append((words, classify))            
    return wordsclassify_set

def nltk_get_words_set(wordsclassify_set):
    words_set = []
    for words, classify in wordsclassify_set:
        words_set.extend(words)
    return words_set

nltk_train_features = None # <- nltk_get_features_set

def nltk_features_func(document):
    document_words = set(document)
    features = {}
    for word in nltk_train_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def nltk_get_features_set(wordsclassify_set):
    freq_dict = nltk.FreqDist(nltk_get_words_set(wordsclassify_set))    
    global nltk_train_features # -> nltk_features_func   
    nltk_train_features = freq_dict.keys()    
    features_set = nltk.classify.apply_features(nltk_features_func, wordsclassify_set)    
    return features_set

def nltk_get_clasifier(train_features_set, method):
    if method == 'nltk_NaiveBayes':   
        classifier = nltk.NaiveBayesClassifier.train(train_features_set)  
    elif method == 'nltk_MaxentGIS':   
        classifier = nltk.MaxentClassifier.train(train_features_set, algorithm='GIS',   trace=1, max_iter=20)

    return classifier

def nltk_classify(test_wordsclassify_set, classifier):
    new_classify_set = [] 
    for words, classify, in test_wordsclassify_set:
        new_classify = classifier.classify(nltk_features_func(words))
        new_classify_set.append(new_classify)
    return new_classify_set
            

# ===========================================================
# sklearn
# ===========================================================

def sklearn_show_most_informative_features(vectorizer, classifier, n=10):
    neg = classifier.feature_log_prob_[0]
    pos = classifier.feature_log_prob_[1]
    valence = (pos - neg)
    ordered = np.argsort(valence)
    #interesting = np.hstack([ordered[:n], ordered[-n:]])
    feature_names = vectorizer.get_feature_names()
    for index in ordered[:n]:
        print "%+4.4f\t%s" % (valence[index], feature_names[index])
    print '\t...'
    for index in ordered[-n:]:
        print "%+4.4f\t%s" % (valence[index], feature_names[index])

def sklearn_get_clasifier(X, y, method):
    if   method == "Svm":   
        classifier = LinearSVC(loss='l1')
    elif method == "Svc":   
        classifier = SVC(kernel='linear', probability=True)
    elif method == "BernoulliNB":   
        classifier = BernoulliNB()
    elif method == "MultinomialNB":   
        classifier = MultinomialNB()
    elif method == "Centroid":   
        classifier = NearestCentroid() # metric = 'manhattan', shrink_threshold=None) #manhattan, euclidian, l2, l1, cityblock
    elif method == "MaxEnt":   
        classifier = LogisticRegression()    
    elif method == "KNeighbors":
        classifier = KNeighborsClassifier(n_neighbors=5,  p=3) # p=1 - manhatnska razdalja, p=2: evklidska; sicer: minkovski   
    #elif method == "DecisionTree":   
    #    classifier = DecisionTreeClassifier()
        
    classifier.fit(X, y)
    
    return classifier

def sklearn_classify(test_vectorizer_array, classifier, min_probability, none_classify):
    
    new_classify_set = classifier.predict(X=test_vectorizer_array)
                    
    if min_probability != None:
        new_probability_set = classifier.predict_proba(X=test_vectorizer_array)
        index = 0
        for new_classify in new_classify_set:
            # če vsaj ena presega min potem je ok (classify je tista ki ima max) 
            ok = False
            for new_probability in new_probability_set[index]:
                if new_probability >= min_probability:
                    ok = True
                    break
            if not ok:
                new_classify_set[index] = none_classify
            index += 1
            
    return new_classify_set

# ===========================================================
# analize
# ===========================================================

def confusion_matrik(y_rocni, y_program):
    cm = confusion_matrix(y_rocni, y_program)
    print cm    
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()

def get_percentage(classify_set, test_new_classify_set):
    count_ok = 0
    index = 0
    for classify in classify_set:
        new_classify = test_new_classify_set[index]
        if new_classify == classify:
            count_ok += 1
        index += 1
    count_all = len(classify_set)
    percentage = float(count_ok) / count_all
    return percentage

def get_analyse(analyses_names, test_data, test_new_classify_set, none_classify):
    
    #is_metrics = False;    
    #for name in analyses_names:
    #    if name in ("precision", "recall", "f1"):
    #        is_metrics = True
    #        break
            
    #if is_metrics:                    
    n_test_classify_set = []
    index = 0
    clear = 0    
    for id, classify, words in test_data:
        if test_new_classify_set[index] != none_classify: 
            n_test_classify_set.append(1 if classify == '+' else 0)
        else:
            clear += 1    
        index = index + 1    
    n_new_classify_set = []
    for new_classify in test_new_classify_set:
        if new_classify != none_classify: 
            n_new_classify_set.append(1 if new_classify == '+' else 0)
    #print clear, len(n_test_classify_set), len(n_new_classify_set)           

    l = len(n_test_classify_set)
    analyse = {}
    for name in analyses_names:
        if l == 0:
            analyse[name] = 0
        elif name == "percentage":
            analyse[name] = get_percentage(n_test_classify_set, n_new_classify_set)
        elif name == "precision":
            analyse[name] = metrics.precision_score(n_test_classify_set, n_new_classify_set)
        elif name == "recall":
            analyse[name] = metrics.recall_score(n_test_classify_set, n_new_classify_set)
        elif name == "f1":
            analyse[name] = metrics.f1_score(n_test_classify_set, n_new_classify_set)  
        #elif name == "confusion":
        #                   confusion_matrik(n_test_classify_set, n_new_classify_set)
    
    return (analyse, clear)


# ===========================================================
# plusi / minusi    
# ===========================================================

def get_plusi_minusi_from_db(db, id):
    cursor = db.cursor()
    cursor.execute("SELECT plusi, minusi FROM komentar WHERE idKomentar = %s" % (id))
    (plusi, minusi) = cursor.fetchone()
    return (plusi, minusi)

def do_plusi_minusi(db, csvfilename): 
    print "\nplusi / minusi"

    #------------------------------
    vsota_plusov_pozitivnih = 0 
    vsota_plusov_negativnih = 0
    vsota_minusov_pozitivnih = 0
    vsota_minusov_negativnih = 0
    stevilom_pozitivnih_komentarjev = 0
    stevilom_negativnoh_komentarjev = 0    
    #------------------------------

    csvfile = open(csvfilename, "rb")
    rows = csv.reader(csvfile, delimiter=';', quotechar='"')
    for id, classify in rows:
        (plusi, minusi) = get_plusi_minusi_from_db(db, id)        
        #------------------------------
        if classify == '+':
            stevilom_pozitivnih_komentarjev += 1
            vsota_plusov_pozitivnih  += int(plusi) 
            vsota_minusov_pozitivnih += int(minusi)
        else:    
            stevilom_negativnoh_komentarjev += 1
            vsota_plusov_negativnih += int(plusi)
            vsota_minusov_negativnih += int(minusi)        
        #------------------------------
    csvfile.close()

    #------------------------------
    print "  Vsota plusov  pozitivnih komentarjev(%i) delejena s številom pozitivnih komentarjev(%i) = %f" % (vsota_plusov_pozitivnih,  stevilom_pozitivnih_komentarjev, float(vsota_plusov_pozitivnih )/stevilom_pozitivnih_komentarjev)
    print "  Vsota plusov  negativnih komentarjev(%i) delejena s številom negativnoh komentarjev(%i) = %f" % (vsota_plusov_negativnih,  stevilom_negativnoh_komentarjev, float(vsota_plusov_negativnih )/stevilom_negativnoh_komentarjev)
    print "  Vsota minusov pozitivnih komentarjev(%i) delejena s številom pozitivnih komentarjev(%i) = %f" % (vsota_minusov_pozitivnih, stevilom_pozitivnih_komentarjev, float(vsota_minusov_pozitivnih)/stevilom_pozitivnih_komentarjev)
    print "  Vsota minusov negativnih komentarjev(%i) delejena s številom negativnoh komentarjev(%i) = %f" % (vsota_minusov_negativnih, stevilom_negativnoh_komentarjev, float(vsota_minusov_negativnih)/stevilom_negativnoh_komentarjev)
    #------------------------------


# ===========================================================
# main
# ===========================================================    

if __name__ == "__main__":

    t0_total = time.time()

    db = MySQLdb.connect("localhost", "root", "lubenica", "mydb", charset="utf8")
    
    # ================================================

    # ----------------------------------------
    # parametri
    # ----------------------------------------

    data_path = "data\\"
    save_data = False    

    csv_path = data_path + "csv\\" 
    all_file = "trainSport.csv"     
    all_csv = csv_path + all_file
    result_path = data_path
    result_csv = data_path + "classify_result[%i](%s %i).csv"
    """  
    data_root = "_random_csv\"
    csv_path = "vsiSport\\"  # "vsi\\", "vsiSport5\\", "vsiSport2\\", "sport\\", "sport2\\", "test2\\", "test3\\", "test4\\"     
    print_log(csv_path)
    csv_path = data_path + data_root + csv_path
    train_csv = csv_path + "train.csv"
    test_csv  = csv_path + "test.csv"
    result_path = csv_path + "result\\"
    result_csv = result_path + "result(%s).csv"
    """
    
    use_nltk = False
    if use_nltk:
        n_iter = 10
        methods = ["nltk_NaiveBayes"] #, "nltk_MaxentGIS"] #, "nltk_MaxentGIS"] #, "nltk_MaxentMegam", "nltk_MaxentIIS", "nltk_Svm", "nltk_DecisionTree"] #, "nltk_KNeighbors"]
        min_probability = None        
        none_classify   = 'N'
        use_corpus      = True
        use_stopwords   = True
        use_neighbors   = False
        min_word_length =  2
        max_word_length = 20
    else:
        #n_iter = 10
        #methods = ["Svc", "Svm", "BernoulliNB", ""MultinomialNB", "Centroid", "MaxEnt", KNeighbors"] #,  "DecisionTree"]
        #min_probability = None
        
        #n_iter = 1
        #methods = ["Svm"]
        #min_probability = None        

        n_iter = 1
        methods = ["Svc"]
        min_probability = 0.7
                
        none_classify   = 'N'        
        use_corpus      = True
        use_stopwords   = True
        use_neighbors   = False
        min_word_length = None
        max_word_length = None
    
    log_file = result_path + "analiza_log.txt"
    analyses_names =  ["percentage", "precision", "recall", "f1"]
    
    print_log(all_file)    

    # ----------------------------------------
    # random split data -> train/test
    # ----------------------------------------
  
    """
    random_path = data_path + "_random\\"
    random_csv       = random_path + "vsiSportR.csv"
    random_train_csv = random_path + "train.csv"
    random_test_csv  = random_path + "test.csv"

    create_csvtrain_and_csvtest(random_csv, random_train_csv, random_test_csv, random_col=5, meja=50)
    print "OK"
    sys.exit()
    """
    
    # ----------------------------------------
    # plusi, minusi
    # ----------------------------------------
    
    """
    train_csv  = csv_path + "trainSport.csv"
    do_plusi_minusi(db, train_csv)
    print "OK"
    sys.exit()
    """
    
    # ----------------------------------------
    # corpus, stop_words
    # ----------------------------------------

    print_log("\ncorpus, stop_words")
    t0 = time.time()

    corpus_path = data_path + "corpus\\"

    if use_corpus:
        corpus = get_corpus(corpus_path + "ssj500kv1_1", corpus_namespace="http://www.tei-c.org/ns/1.0")
    else:    
        corpus = None
    
    if use_stopwords: 
        stop_words = get_stopwords(corpus_path + "stopwords_slo", corpus)
    else:
        stop_words = None

    print_log("  čas = %i ms" % ((time.time() - t0) * 1000))
    
    
    # ----------------------------------------
    # data
    # ----------------------------------------
    
    all_data = load_data_from_csv("all", db, all_csv, corpus, min_word_length, max_word_length, stop_words, use_neighbors)    
    
    
    # ----------------------------------------
    # vectorizer
    # ----------------------------------------
    
    if use_nltk:
        vectorizer = None
    else:    
        vectorizer  = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=1, max_df=80)
        #vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,1), min_df=1, binary=True)


    # ==================================================================
    # analiza - iteracije
    # ==================================================================
    
    analyses_set = []
    
    print_log("\ndata = %i" % len(all_data))
    print_log("iter = %i" % n_iter)
    t0_analize = time.time()

    iter_index = 0
    ss = ShuffleSplit(n=len(all_data), n_iter=n_iter, test_size=0.25, indices=True)    
    for train_index, test_index in ss:        
        iter_index += 1
        t0_iter = time.time()


        # ----------------------------------------
        # split data -> train/test
        # ----------------------------------------
        
        train_data = get_index_data(train_index, all_data)
        test_data  = get_index_data(test_index,  all_data)


        # ----------------------------------------
        # train - classifiers
        # ----------------------------------------
        
        if use_nltk:
            train_wordsclassify_set = nltk_get_wordsclassify_set(train_data)
            train_features_set      = nltk_get_features_set(train_wordsclassify_set)
        else:
            train_words_set         = get_words_set(train_data)
            train_classify_set      = get_classify_set(train_data)        
            train_vectorizer_array  = vectorizer.fit_transform(train_words_set).toarray()
        
        classifiers = {}
        for method in methods:
            if use_nltk:
                classifiers[method] = nltk_get_clasifier(train_features_set, method)
                #print "  accuracy = %i%%" % nltk.classify.accuracy(classifiers[method], train_features_set) * 100
                #print classifiers[method].show_most_informative_features(25)        
            else:
                classifiers[method] = sklearn_get_clasifier(X=train_vectorizer_array, y=train_classify_set, method=method)
                #sklearn_show_most_informative_features(vectorizer, classifiers[method], n=25)  # <- ne dela za "Centroid" in "MaxEnt"


        # ----------------------------------------
        # test - new classify
        # ----------------------------------------
        
        if use_nltk:
            test_wordsclassify_set = nltk_get_wordsclassify_set(test_data)
        else:
            test_words_set = get_words_set(test_data)
            test_vectorizer_array = vectorizer.transform(test_words_set).toarray()
        
        test_new_classify_sets = {}
        for method in methods:
            if use_nltk:
                test_new_classify_sets[method] = nltk_classify(test_wordsclassify_set, classifiers[method])
            else:  
                test_new_classify_sets[method] = sklearn_classify(test_vectorizer_array, classifiers[method], min_probability, none_classify)
        
        # ----------------------------------------
        # analiza
        # ----------------------------------------
                
        analyses = {}
        sum_clear = 0
        for method in methods:
            test_new_classify_set = test_new_classify_sets[method]
            (analyses[method], clear) = get_analyse(analyses_names, test_data, test_new_classify_set, none_classify)
            sum_clear += clear
        analyses_set.append(analyses)
        
        if min_probability != None:
            print "       probability = %.2f clean = %i/%i" % (min_probability, sum_clear, len(test_new_classify_set))
            
        print "  [%i] čas = %i ms" % (iter_index, (time.time() - t0_iter) * 1000)


        # ----------------------------------------
        # save
        # ----------------------------------------
        
        if save_data:
            for method in methods:
                test_new_classify_set = test_new_classify_sets[method]
                save_data_notok_to_csv(   db, test_data, test_new_classify_set, result_csv % (iter_index, method, len(test_new_classify_set)), none_classify)
                #save_data_to_csv(        db, test_data, test_new_classify_set, result_csv % (iter_index, method, len(test_new_classify_set))
                #save_new_classify_to_csv(db, test_data, test_new_classify_set, result_csv % (iter_index, method, len(test_new_classify_set))

        
    print_log("  čas = %i ms" % ((time.time() - t0_analize) * 1000))

    # ==================================================================


    # ----------------------------------------
    # analiza - izračun povprečja
    # ----------------------------------------

    analyses_avg = {}
        
    for method in methods:
        analyses_avg[method] = {}
        for name in analyses_names:
            analyses_avg[method][name] = 0
              
    for analyses in analyses_set:
        for method in methods:
            for name in analyses_names:
                analyses_avg[method][name] += analyses[method][name]
    
    count_all = len(analyses_set)
    for method in methods:
        for name in analyses_names:
            analyses_avg[method][name] /= count_all


    # ----------------------------------------
    # analiza - izpis 
    # ----------------------------------------
    
    if min_probability != None:
        print_log("\nanaliza (probability = %.2f)" % min_probability)
    else:     
        print_log("\nanaliza")
        
    print_log("  %s %s" % ("method", analyses_names))
    for method in methods:
        analyses = ""
        for name in analyses_names:            
            analyses += ", %i%%" % (analyses_avg[method][name] * 100)
        analyses = analyses[2:] # pobrišemo prvo vejico ", "   
        print_log("  %s [%s]" % (method, analyses))

    # ----------------------------------------

        
    db.close()
    
    print_log("\nskupni čas = %i ms" % ((time.time() - t0_total) * 1000))
    print_log("OK")

    write_log(log_file)

