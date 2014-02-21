from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
#from sklearn.cluster import Ward

fig = plt.figure()
fig.clf()
ax = Axes3D(fig)
st = PorterStemmer()

useless = {}
useless[ord('.')] = ""
useless[ord(',')] = ""
useless[ord('\'')] = None
useless[ord('"')] = None
useless[ord('!')] = ""
useless[ord('?')] = "" 
useless[ord('-')] = "" 
useless[ord('!')] = "" 
useless[ord('#')] = None
useless[ord('1')] = None
useless[ord('2')] = None
useless[ord('3')] = None
useless[ord('4')] = None
useless[ord('5')] = None
useless[ord('6')] = None
useless[ord('7')] = None
useless[ord('8')] = None
useless[ord('9')] = None
useless[ord('0')] = None
useless[ord('(')] = ""
useless[ord(')')] = ""

def meaningful(s):
    fillers = set(['an', 'or', 'make', 'for', 'a', 'as', 'the', 'use', 'if', 'of', 'in',
                   'it', 'to', 'with', 'this'])
    if s not in fillers:
        return s

def getFeatures(s):
    for i in range(len(s)-2):
        yield(s[i:i+2])
    for i in range(len(s)-3):
        yield(s[i:i+3])
    for i in range(len(s)-4):
        yield(s[i:i+4])
    for word in s.split():
        yield word

data_file = open('cm_combined_alt_uses.txt', 'r')

features = set()
#lines = [" ".join(filter(meaningful, [st.stem(a) for a in
#                                      line.lower().translate(useless).split()]))
#         for line in data_file if line.strip() != ""]
lines = [" ".join(filter(meaningful, line.lower().translate(useless).split()))
         for line in data_file if line.strip() != ""]
lines = list(set(lines))
lines = [line for line in lines if len(line) < 23]

for line in lines:
    for f in getFeatures(line):
        features.add(f)

features = list(features)

vectors = []

for line in lines:
    vector = []
    for feature in features:
        if feature in getFeatures(line):
            vector.append(1 / len(line.split()))
        else:
            vector.append(0)
    vectors.append(vector)

pca = PCA(n_components=1)
distances = pca.fit_transform(vectors)
pairs = [a for a in zip(distances, lines)]
pairs.sort()
for a in pairs:
    print(a[1])

lengths = [len(line) for line in lines]
mean = sum(lengths)/len(lengths)
diffs = [abs(mean - len(line)) for line in lines]
std = sum(diffs)/len(diffs)
print(mean)
print(std)

#pca = PCA(n_components=2)
#points = pca.fit_transform(vectors)
#X,Y = zip(*points)
#plt.plot(X,Y, 'ro')
#plt.show()

pca = PCA(n_components=3)
points = pca.fit_transform(vectors)
X,Y,Z = zip(*points)
ax.scatter(X,Y,Z)
plt.draw()
plt.show()

#for line in lines:
#    print(line)

data_file.close()

