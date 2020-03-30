import re
import numpy as np
import nltk
import os
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')  
def upload():  
    return render_template("inputFile.html")

@app.route('/output', methods = ['POST'])
def plagiarismDetection():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
    #fname = input("Enter the file name: ")

    # Reading the text file

        fo=open(f.filename,"rb")
        item=fo.read()
        #print(item)
        fo.close()

        #Text segmentation (sentence wise, removing all trailing and leading spaces and blank and null entries)

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


        for ele in segments: 
            text = nltk.word_tokenize(ele)
            count=len(text)
            wordCount.append(count)
            posTagged = pos_tag(text)
            simplifiedTags = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in posTagged]
            listOfPartsOfSpeechWithWords.append(simplifiedTags)
        print(listOfPartsOfSpeechWithWords)

        featuresHeading=['sentence number ','PRON','.','word count','DET']


        #this cell is for adding features heading in dataTable
        length=len(featuresHeading)
        dataTable=np.zeros([0,length],dtype = int)



        i=1

        for i in range(len(listOfPartsOfSpeechWithWords)): #ele in listOfPartsOfSpeechWithWords:
            rowToAdd=np.zeros(length, dtype= float)
            rowToAdd[0]=i+1
            #i+=1
            j=0
            for n in listOfPartsOfSpeechWithWords[i]:
                checkFeature=n[1]
            
                #print(checkFeature)
                for feature in featuresHeading:
                    if j>=length:
                        j=0
                    
                    
                    if(checkFeature==feature):
                        rowToAdd[j]+=1
                    j+=1
                    
            rowToAdd[3]= wordCount[i]/10        
            dataTable=np.vstack((dataTable,rowToAdd))
        print(dataTable)


        from copy import copy, deepcopy
        new = deepcopy(dataTable)


        new=np.delete(new,0,axis=1)


        kmeans=KMeans(n_clusters=5).fit(new)

        labels=list(kmeans.labels_)
        print(labels)
        #print(kmeans.cluster_centers_)

        #plt.figure(figsize=(10,10))

        #plt.subplot(2,2,1)
        plt.figure(num=0)
        plt.scatter(new[:,2],new[:,0], c=kmeans.labels_, cmap='rainbow')
        plt.title('Pronoun')
        plt.xlabel('sentence length')
        plt.ylabel('pronoun')
        plt.scatter(kmeans.cluster_centers_[:,2] ,kmeans.cluster_centers_[:,0], color='black')
        plt.savefig('static/fig0.png')

        #plt.subplot(2,2,2)
        plt.figure(num=1)
        plt.scatter(new[:,2],new[:,1], c=kmeans.labels_, cmap='rainbow')
        plt.title('Punctuation')
        plt.xlabel('sentence length')
        plt.ylabel('punctuation')
        plt.scatter(kmeans.cluster_centers_[:,2] ,kmeans.cluster_centers_[:,1], color='black')
        plt.savefig('static/fig1.png')

        #plt.subplot(2,2,3)
        plt.figure(num=2)
        plt.scatter(new[:,2],new[:,3], c=kmeans.labels_, cmap='rainbow')
        plt.title('Determiner')
        plt.xlabel('sentence length')
        plt.ylabel('determiner')
        plt.scatter(kmeans.cluster_centers_[:,2] ,kmeans.cluster_centers_[:,3], color='black')
        plt.savefig('static/fig2.png')

        #plt.tight_layout()
        #plt.show()



        #plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')



        #countiing the frequency
        zero=one=two=three=four=0
        for ele  in labels:
            if(ele==0):
                zero+=1
            if(ele==1):
                one+=1
            if(ele==2):
                two+=1
            if(ele==3):
                three+=1
            if(ele==4):
                four+=1
        print(zero,one,two,three,four)

        max=0
        maxf=0
        if(max<zero):
            max=zero
            maxf=0
        if(max<one):
            max=one
            maxf=1
        if(max<two):
            max=two
            maxf=2
        if(max<three):
            max=three
            maxf=3
        if(max<four):
            max=four
            maxf=4
        print(max,maxf)



        i=0
        for ele in labels: 
            
            if(ele!=maxf):
                print(segments[i])
            i+=1
    return render_template("output.html", segments=segments, labels=labels, maxf = maxf, fig0 = '/static/fig0.png', fig1 = '/static/fig1.png', fig2 = '/static/fig2.png')

if __name__ == '__main__':
    app.run()