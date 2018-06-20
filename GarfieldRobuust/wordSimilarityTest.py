import gensim
print("start reading in ...")
modelNLP = gensim.models.KeyedVectors.load_word2vec_format('/home/axel/Documents/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

words = [['back', 'next'],['marker', 'house'], ['back', 'marker'], [
    'next', 'marker'],['house', 'building'],['house', 'room'] , ['house', 'home'] , ['marker', 'pen'], ['marker', 'badge'] , ['next', 'succeed'], ['back', 'cancel']
]


for i in range(0, len(words)):
    print("We will test similarity between:", words[i][0], " and ", words[i][1] , " : ", modelNLP.similarity(words[i][0], words[i][1]) )
    print(modelNLP.wv[words[i][0]][:10], modelNLP.wv[words[i][1]][:10])



