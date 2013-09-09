# -*- coding: utf-8 -*-

"""
klasifikacija.py 
"""

import time
import os
import re
import csv
import MySQLdb
import xml.etree.cElementTree as et

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# ===========================================================
# print log 
# ===========================================================

logs = []
def print_log(msg):
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
    print_log("  stopwords = %i " % len(stop_words))    
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
# load db
# ===========================================================

def get_sport_data_from_db(db, offset, limit):
    cursor = db.cursor()
    cursor.execute("SELECT komentar.idKomentar, komentar.besedilo FROM komentar, novica " + \
                   "WHERE novica.kategorija = 'sport' AND komentar.Novica_idNovica = novica.idNovica AND " + \
                   "TRIM(komentar.besedilo) != '' AND komentar.besedilo NOT LIKE '@%%' LIMIT %i,%i" % (offset, limit))  
    rows = cursor.fetchall()
    return rows

def load_data_from_db(msg, db, offset, limit, corpus, min_word_length, max_word_length, stop_words, use_neighbors): 
    t0 = time.time()
    print_log("\nload %s [%i,%i]" % (msg, offset, limit))
    rows = get_sport_data_from_db(db, offset, limit)
    print_log("  load = %i ms" % ((time.time() - t0) * 1000))

    t0 = time.time()
    data = []
    words_count = 0
    clean_count = 0
    replace_count = 0
    for id, besedilo in rows:
        words = get_words_from_besedilo(besedilo)
        words_count += len(words) 
        (words, clean, replace) = get_clean_words(words, corpus, min_word_length, max_word_length, stop_words, use_neighbors)
        if len(words) > 0: 
            clean_count += clean
            replace_count += replace
            data.append((id, None, words))

    if corpus != None:
        if words_count > 0:
            replace_procent = replace_count * 100 / words_count
        else:
            replace_procent = 0                    
        print_log("  corpus = %i%% %i/%i" % (replace_procent, replace_count, words_count))
    if clean_count > 0:            
        print_log("  clean(stop,min,max) = %i" % clean_count)                

    print_log("  data = %i ms" % ((time.time() - t0) * 1000))
    return data

# ===========================================================
# save
# ===========================================================    

def writerow_to_db(db, id, classify):
    cursor = db.cursor()
    cursor.execute("INSERT INTO klasifikacije (id, classify) VALUES (%i, '%s') " + \
                   "ON DUPLICATE KEY UPDATE classify = '%s'" % (id, classify, classify))
    
def save_data_to_db(db, test_data, test_new_classify_set):
    t0 = time.time()
    row = 0
    for id, classify, words in test_data:
        new_classify = test_new_classify_set[row]
        # ------------    
        writerow_to_db(db, id, new_classify)
        # ------------    
        row += 1
    db.commit()
    print_log("  save to db = %i ms" % ((time.time() - t0) * 1000))

# ===========================================================
# data
# ===========================================================

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
# sklearn
# ===========================================================

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
    train_file = "trainSport.csv"     
    #test_file = "testSport.csv"     
    train_csv = csv_path + train_file
    #test_csv = csv_path + test_file
    result_path = data_path
    result_csv  = data_path + "klasifikacija(%s %ix%i)[%i,%i].csv"
    
    #method         = "Svm"
    #min_probability = None    
    method          = "Svc"
    min_probability = 0.7
    
    use_neighbors   = False
    min_word_length = None
    max_word_length = None

    none_classify   = 'N'
    
    print_log(train_file)    
    print_log(method)
    log_file = result_path + "klasifikacija_log.txt"
    
    
    # ----------------------------------------
    # corpus, stop_words
    # ----------------------------------------

    print_log("\ncorpus, stop_words")
    t0 = time.time()

    corpus_path = data_path + "corpus\\"
    corpus      = get_corpus(corpus_path + "ssj500kv1_1", corpus_namespace="http://www.tei-c.org/ns/1.0")
    stop_words  = get_stopwords(corpus_path + "stopwords_slo", corpus)
    
    print_log("  čas = %i ms" % ((time.time() - t0) * 1000))

    # ----------------------------------------
    # vectorizer
    # ----------------------------------------
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=1)

    # ---------------------------------    
    # train - classifiers
    # ---------------------------------    
    train_data = load_data_from_csv("train", db, train_csv, corpus, min_word_length, max_word_length, stop_words, use_neighbors)    

    train_words_set    = get_words_set(train_data)
    train_classify_set = get_classify_set(train_data)        
    train_vectorizer_array = vectorizer.fit_transform(train_words_set).toarray()    
    classifier = sklearn_get_clasifier(X=train_vectorizer_array, y=train_classify_set, method=method)

    
    # ================================================
    limit  = 1000    
    offset = 0
    while True:
        t0_limit = time.time()

        # ---------------------------------    
        # test data
        # ---------------------------------    
        #test_data = load_data_from_csv("test",  db, test_csv,      corpus, min_word_length, max_word_length, stop_words, use_neighbors)
        test_data  = load_data_from_db( "test",  db, offset, limit, corpus, min_word_length, max_word_length, stop_words, use_neighbors)

        if len(test_data) == 0:
            break
        
        # ---------------------------------    
        # test - new classify
        # ---------------------------------    
        print_log("  classify %i/%i" % (len(train_data), len(test_data)))
        t0 = time.time()
    
        test_words_set = get_words_set(test_data)
        test_vectorizer_array = vectorizer.transform(test_words_set).toarray()
        
        test_new_classify_set = sklearn_classify(test_vectorizer_array, classifier, min_probability, none_classify)   
        #test_new_classify_set = classifier.predict(X=test_vectorizer_array)

        print_new_classify_set(test_new_classify_set)
        print_log("  classify = %i ms" % ((time.time() - t0) * 1000))
        
        # ---------------------------------    
        # save
        # ---------------------------------    
        print_log("  data = %i" % len(test_data))
        if save_data:
            save_data_to_db(db, test_data, test_new_classify_set)
        
        # ---------------------------------    

        print_log("  čas = %i ms / %i ms" % ((time.time() - t0_limit) * 1000, (time.time() - t0_total) * 1000))
        offset += limit
    # ================================================

    db.close()
        
    print_log("\nskupni čas = %i ms" % ((time.time() - t0_total) * 1000))
    print_log("OK")
    
    write_log(log_file)
