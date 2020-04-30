import re
import numpy as np
import nltk
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
from nltk.corpus import stopwords
import string
import matplotlib.colors as mc
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')  
def upload():  
    return render_template("inputFile.html")

@app.route('/about')  
def about():  
    return render_template("aboutUs.html")

@app.route('/output', methods = ['POST'])
def plagiarismDetection():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
    #fname = input("Enter the file name: ")

    # Reading the text file

        fo=open(f.filename,"r")
        item=fo.read()
        #print(item)
        fo.close()

        #Text segmentation (sentence wise, removing all blank and null entries)

        segments=re.split(r'\. |\n',item)
        print(segments)

        def space(ele): 
            if ele.isspace() or not ele: 
                return False
            else: 
                return True
        segments = list(filter(space, segments))
        print(segments)

        from nltk.tag import pos_tag, map_tag
        listOfPartsOfSpeechWithWords=[]
        wordCount=[]
        tokens=[]
        wordFrequency={}
        wordFrequencySentenceWise=[]
        maxFreqWord=''
        maxFreq=0
        wordFrequencyRatio=[]


        for ele in segments: 
            text = nltk.word_tokenize(ele)
            tokens.append(text)
            count=len(text)
            wordCount.append(count)
            posTagged = pos_tag(text)
            simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
            listOfPartsOfSpeechWithWords.append(simplifiedTags)
        print("list of tokens with tags: ", listOfPartsOfSpeechWithWords)

        featuresHeading=['sentence number ','PRON','.','word count','DET','word frequency ratio']

        #set of stop words
        stop_words = set(stopwords.words('english')).union(string.punctuation)
        print("stop words: ",stop_words)

        # word Frequency
        for row in tokens:
            wfs={}
            for word in row:
                word1=word.lower()
                if word1 not in stop_words:
                    wordFrequency[word1]=wordFrequency.get(word1, 0) + 1
                    wfs[word1]=wfs.get(word1, 0)+1
            wordFrequencySentenceWise.append(wfs)

        print("wordFrequencySentenceWise: ", wordFrequencySentenceWise)
        print("word Frequency: ", wordFrequency)
        maxFreqWord=max(wordFrequency,key = wordFrequency.get)
        maxFreq=wordFrequency[maxFreqWord]
        print("max frequency word: ", maxFreqWord)
        print("max frequency: ", maxFreq)


        # creating dataTable
        length=len(featuresHeading)
        dataTable=np.zeros([0,length],dtype = int)

        from statistics import mean
        import math

        for i in range(len(listOfPartsOfSpeechWithWords)): #ele in listOfPartsOfSpeechWithWords:
            rowToAdd=np.zeros(length, dtype= float)
            wfrs={}
            meanwfrs = 0
            rowToAdd[0]=i+1
            j=0
            for n in listOfPartsOfSpeechWithWords[i]:
                checkFeature=n[1]
                word=n[0].lower()
                for feature in featuresHeading:
                    if j>=length:
                        j=0
                    if(checkFeature==feature):
                        rowToAdd[j]+=1
                    j+=1
                if word not in stop_words:
                    wfrs[word]= math.log2(maxFreq/(wordFrequency[word]-wordFrequencySentenceWise[i][word]+1))
            if bool(wfrs):
                meanwfrs = mean(wfrs[k] for k in wfrs)
            wordFrequencyRatio.append(wfrs)
        
            rowToAdd[3]= wordCount[i]/10 
            rowToAdd[5]= meanwfrs       
            dataTable=np.vstack((dataTable,rowToAdd))
        print("Data Table: ", dataTable)


        from copy import copy, deepcopy
        new = deepcopy(dataTable)
        new=np.delete(new,0,axis=1)
        updatedDataTable = np.delete(new,3,axis=1)

        # Compute DBSCAN
        X= updatedDataTable
        db = DBSCAN(eps=1, min_samples=3).fit(X)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask= np.zeros(len(db.labels_),dtype=bool)
        for ele in db.core_sample_indices_:
            core_samples_mask[ele] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters)
        print('Estimated number of noise points: %d' % n_noise)

        # Plot result
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        clusterFrequency= np.zeros(n_clusters,dtype='int')
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        labelColour = list(zip(unique_labels, colors))
        
        for k, col in labelColour:
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)
            if(k!=-1):
                clusterFrequency[k]=np.count_nonzero(class_member_mask)

            xc = X[class_member_mask & core_samples_mask]
            xnc = X[class_member_mask & ~core_samples_mask]
            
            fig0=plt.figure(num=0)
            plt.title('Pronoun')
            plt.xlabel('sentence length')
            plt.ylabel('pronoun')
            
            plt.plot(xc[:, 2], xc[:, 0], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            plt.plot(xnc[:, 2], xnc[:, 0], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            plt.savefig('static/fig0.png')
            
            
            fig1=plt.figure(num=1)
            plt.title('Punctuation')
            plt.xlabel('sentence length')
            plt.ylabel('punctuation')
            
            plt.plot(xc[:, 2], xc[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            plt.plot(xnc[:, 2], xnc[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            plt.savefig('static/fig1.png')

            '''
            fig2=plt.figure(num=2)
            plt.title('Determiner')
            plt.xlabel('sentence length')
            plt.ylabel('determiner')
            
            plt.plot(xc[:, 2], xc[:, 3], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            plt.plot(xnc[:, 2], xnc[:, 3], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            plt.savefig('static/fig2.png')
            '''

            fig2=plt.figure(num=2)
            plt.title('Word Frequency Ratio')
            plt.xlabel('sentence length')
            plt.ylabel('word frequency ratio')
            
            plt.plot(xc[:, 2], xc[:, 3], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            plt.plot(xnc[:, 2], xnc[:, 3], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
            plt.savefig('static/fig2.png')
            print("col:",col,tuple(col))

        fig0.clf()
        fig1.clf()
        fig2.clf()
        #fig3.clf()
        if(n_clusters!=0):
            maxFreqCluster=np.argmax(clusterFrequency)
        else:
            maxFreqCluster= 0
        
        print("Noise: ")
        for i in range(len(labels)):
            if(labels[i]==-1):
                print(segments[i])

        colours=[]
        for i in range(len(labels)):
            if labels[i]==maxFreqCluster:
                colours.append("#FFFFFF")
            elif labels[i] == -1:
                colours.append("#808080")
            else:
                for k,col in labelColour:
                    if labels[i]==k:
                        colours.append(mc.to_hex(col))


    #return render_template("output.html", segments=segments, labels=labels, maxf = maxf, fig0 = '/static/fig0.png', fig1 = '/static/fig1.png', fig2 = '/static/fig2.png')
    return render_template("output.html",segments=segments, labels=labels,labelColour = labelColour, colours = colours, maxFreqCluster=maxFreqCluster,fig0 = '/static/fig0.png', fig1 = '/static/fig1.png', fig2 = '/static/fig2.png', n_clusters = n_clusters )
if __name__ == '__main__':
    app.run()

"""# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response"""