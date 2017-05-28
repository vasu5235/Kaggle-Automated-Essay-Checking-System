from requirements import *

#Helper functions for preprocessing, generating custome heuristic features and Word2Vec features

def preprocess_data2feature(X,target_col):
    # Generate feature columns for 15 custom features and a target value column    
    feature_columns = ["essay","word_count","long_word_count","avg_word_length_per_essay","wrong_words","no_of_domain_words","word_to_sent_ratio","num_of_characters","sentence_count","noun_count","verb_count","comma_count","punctuation_count","adjective_count","adverb_count","quotation_mark_count","spelling_mistakes","target"]
    feature_pd = pd.DataFrame(index = X.index, columns = feature_columns)
    feature_pd['essay'] = X['essay']
    feature_pd['target'] = X[target_col]
    
    return feature_pd

def featureSet2(X): 
    # Extract features from the given essay and assign the value/count to the respective column.
    for index,row in X.iterrows():
        
        text = unicode(row['essay'],errors='ignore') 
        text = " ".join(filter(lambda x:x[0]!='@', text.split())) #To remove proper nouns tagged in the data-set which may result into false positives during POS tagging.
        
        punctuation = ['.','?', '!', ':', ';']
        #Comma count
        comma_count = text.count(',')
        row['comma_count'] = comma_count
        
        #Punctuation count
        punctuation_count = 0
        for punct in punctuation:
            punctuation_count += text.count(punct)
        row['punctuation_count'] = punctuation_count
        
        #Quotation marks count
        quotation_mark_count = text.count('"')
        quotation_mark_count += text.count("'")
        row['quotation_mark_count'] = quotation_mark_count
        
        #Add the sentence count
               
        tokenized_essay = nltk.sent_tokenize(text)
        sent_count = len(tokenized_essay)
        row['sentence_count'] = sent_count
        
        #Add word count after removing the stop words.
        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 
        
        for word in words:
            if word in stop_words:
                words.remove(word)
        word_count = len(words)
        
        row['word_count'] = word_count
        
        #Long word count
        long_word_count = 0
        total_word_length = 0
        for word in words:
            total_word_length += len(word)
            if len(word) > 6:
                long_word_count +=1
        row['long_word_count'] = long_word_count
        
        #Average word length per essay
        row['avg_word_length_per_essay'] = round((total_word_length/float(len(words))),2)
        
        
        tool = grammar_check.LanguageTool('en-US')
        matches = tool.check(text)
        row['spelling_mistakes'] = len(matches)
           
        #POS TAGS
        count= Counter([j for i,j in nltk.pos_tag(words)])
               
        row['noun_count'] = count['NN'] + count['NNS'] + count['NNPS'] + count['NNP']
        row['verb_count'] = count['VB'] + count['VBG'] + count['VBP'] + count['VBN'] + count['VBZ']
        row['adjective_count'] = count['JJ'] + count['JJR'] 
        row['adverb_count'] = count['RB'] + count['RBR'] + count['RBS']
        
        #No_of_domain_words and wrong words after removing the stop words and punctuations from the essay.
        cnt = 0
        wrong_word_count = 0
        for word in words:
            if wn.synsets(word):
                cnt += 1
            else:
                wrong_word_count += 1
        row['no_of_domain_words'] = cnt
        row['wrong_words'] = wrong_word_count        
        
        #Word to sentence ratio
        row['word_to_sent_ratio'] = round(float(word_count/float(sent_count)),2)
        
        #Number of characters
        row['num_of_characters'] = nltk.FreqDist(text).N()
        
        #Debugging
        if index%10==0:
            print "made features for rows with index upto ",index
        
def GenerateFeatures(X):
    start = time()
    featureSet2(X)
    end = time()
    print ("Generated the features for the entire data-set in {:.4f} minutes".format((end - start)/60.0))

#Fitting, predicting and calculating error. 
#Using LinearRegression, 5 fold cross validation and quadratic kappa as an error metric.

def Evaluate(X_all,y_all,feature_list):
    
    #Fitting, predicting and calculating error. 
    #Using LinearRegression, 5 fold cross validation and quadratic kappa as an error metric.
    
    model = LinearRegression()

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(X_all), n_folds=5,shuffle=True)
    results = []
    
    for traincv, testcv in cv:
            X_test, X_train, y_test, y_train = X_all.iloc[testcv], X_all.iloc[traincv], y_all.iloc[testcv], y_all.iloc[traincv]
                     
            final_train_data = X_train[feature_list]
            final_test_data = X_test[feature_list]
                     
            model.fit(final_train_data,y_train)
            start = time()
            y_pred = model.predict(final_test_data)
            end = time()
                                   
            result = kappa(y_test.values,y_pred,weights='quadratic')
            results.append(result)
            
            X_test_list = [i for i in range(len(X_test))]
            plt.scatter(X_test_list,y_test.values,color='black')
            plt.scatter(X_test_list,y_pred,color='blue')
            
    #print "Results: " + str( np.array(results).mean() )
    return str(np.array(results).mean())

#Word2Vec modules
def essay_to_wordlist(essay_v, remove_stopwords):
    # Remove the tagged labels and word tokenize the sentence.
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    # Sentence tokenize the essay and call essay_to_wordlist() for word tokenization.
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    # Helper function for generating word vectors.
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])        
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    #Main function to generate the word vectors for word2vec model.
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32") # len(essays) X num_features matrix
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

#Word2Vec modules ends
