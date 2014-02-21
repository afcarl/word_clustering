from lsaUC import lsa_uc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
from nltk.metrics.distance import edit_distance
from sklearn.cluster import KMeans
from sklearn.cluster import Ward
import numpy as np
import pickle

def init():
    tokenizer = RegexpTokenizer(r'[a-z]+')
    #data_file = open('cm_combined_alt_uses.txt', 'r')
    data_file = open('all-uses.txt', 'r')

    features = set()
    lines = []
    for line in data_file:
        if line.strip() != "":
            words = []
            for word in tokenizer.tokenize(line.lower()):
                words.append(word)
                if word not in stopwords.words('english'):
                    features.add(word)
            lines.append(" ".join(words))

    #lines = list(set(lines))
    #limit the number for now TEMP
    #lines = lines[0:100]
    #print(lines)
    #features = set()
    #for line in lines:
    #    for word in tokenizer.tokenize(line.lower()):
    #        if word not in stopwords.words('english'):
    #            features.add(word)
    #print(features)
    #END temp
    #print(len(lines))
    data_file.close()
    return features, lines


def get_X(lines, features, cache):
    if cache == None:
        cache = {}
    tokenizer = RegexpTokenizer(r'[a-z]+')
    X = []
    for line1 in lines:
        vector = []
        for line2 in lines:
            vector.append(edit_distance(line1,line2)/max(len(line1),len(line2)))
        max_v = max(vector)
        for i in range(len(vector)):
            vector[i] = vector[i] / max_v 
        syn_dist = {}
        for word in features:
            syn_dist[word] = 1

        for word in set(tokenizer.tokenize(line1.lower())):
            if word in stopwords.words('english'):
                continue

            for word2 in features:
                if (len(wn.synsets(word)) == 0 or len(wn.synsets(word2)) == 0):
                    continue
                else:
                    if (word not in cache):
                        cache[word] = {}
                    if (word2 not in cache[word]):
                        similarity = [w1.wup_similarity(w2)
                                      for w1 in wn.synsets(word, pos=wn.NOUN) 
                                      + wn.synsets(word, pos=wn.VERB)
                                      for w2 in wn.synsets(word2, pos=wn.NOUN)
                                      + wn.synsets(word2, pos=wn.VERB)]
                        similarity = [s for s in similarity if s]

                        if (len(similarity) != 0):
                            cache[word][word2] = max(similarity)
                        else:
                            cache[word][word2] = None

                        #cache[word][word2] = wn.synsets(word)[0].path_similarity(wn.synsets(word2)[0])
                        #cache[word][word2] = wn.synsets(word)[0].wup_similarity(wn.synsets(word2)[0])

                    if (not cache[word][word2]):
                        continue

                    dist = 1 - cache[word][word2]
                    if (dist < syn_dist[word2]):
                        syn_dist[word2] = dist 
                        
        for word in features:
            vector.append(syn_dist[word])

        X.append(vector)
    return X, cache

#print(X)        
#print(list(set(lines)))
#print(features)


# PCA to create a sorted list


# PCA to plot in 2d
#pca = PCA(n_components=2)
#points = pca.fit_transform(vectors)
#X,Y = zip(*points)
#plt.plot(X,Y, 'ro')
#plt.show()

# PCA to plot in 3d
#pca = PCA(n_components=3)
#points = pca.fit_transform(X)
#x,y,z = zip(*points)
#ax.scatter(x,y,z)
#plt.draw()
#plt.show()

def print_sorted(X,lines):
    pca = PCA(n_components=1)
    distances = pca.fit_transform(X)
    pairs = [a for a in zip(distances, lines)]
    pairs.sort()
    for a in pairs:
        print(a[1])

def cluster(X, lines, k):
    if not k:
        k=60
    pca = PCA(n_components=0.99)
    points = pca.fit_transform(X)
    clust = KMeans(init='k-means++', n_clusters=k, n_init=30)
    #clust = Ward(n_clusters=k)
    clust.fit(points)


    for c in range(0,k):
        print()
        print("CLUSTER #%i" % c)
        for i,line in enumerate(lines):
            #if clust.labels_[i] == c:
            if clust.predict(points[i]) == c:
                print(line)

def main():
    features, lines = init()

    pickle.dump(lines, open('uses.p', 'wb'))
    similarity_matrix = {}
    print(lines)
    for phrase in lines:
        for phrase2 in lines:
            print("%s - %s" % (phrase, phrase2))
            if phrase not in similarity_matrix:
                similarity_matrix[phrase] = {}
            if phrase2 not in similarity_matrix:
                similarity_matrix[phrase2] = {}
            if phrase2 not in similarity_matrix[phrase]:
                similarity = lsa_uc(phrase, phrase2)
                similarity_matrix[phrase][phrase2] = similarity
                similarity_matrix[phrase2][phrase] = similarity

    pickle.dump(similarity_matrix, open('similarity.p', 'wb'))

    #print(similarity_matrix)

if __name__ == "__main__":
    main()
