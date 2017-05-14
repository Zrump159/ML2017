import word2vec
import numpy as np
import nltk

#word2vec.word2phrase("all.txt","all-phrase",verbose=True)
#word2vec.word2vec('all-phrase','all.bin',size=120,verbose=True)
#word2vec.word2vec('all.txt','all.bin',size=150,window=5,min_count=4,sample=1e-5,verbose=True)
'''
  word2vec.word2vec(
        train=args.corpus_path,
        output=args.model_path,
        cbow=MODEL,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
verbose=True)

word2vec(train, output, size=100, window=5, sample='1e-3', hs=0,
        negative=5, threads=12, iter_=5, min_count=5, alpha=0.025,
        debug=2, binary=1, cbow=1, save_vocab=None, read_vocab=None,
        verbose=False):
'''
#word2vec.word2clusters('all.txt', 'all-clusters.txt', 100, verbose=True)


model = word2vec.load('all.bin')

vocabs = []
vecs = []
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
vocabs = vocabs[:800]
vecs = np.array(vecs)[:800]


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)

import matplotlib.pyplot as plt
from adjustText import adjust_text

# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!"]

texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

# plt.savefig('hp.png', dpi=600)
plt.show()






