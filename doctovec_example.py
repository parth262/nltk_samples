from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

print(common_texts)

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

print(documents)

model = Doc2Vec(documents, size=5, window=2, min_count=1, workers=4)

from gensim.test.utils import get_tmpfile
fname = get_tmpfile('my_doc2vec_model')

print(fname)

model.save(fname)
model = Doc2Vec.load(fname)

model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

vector = model.infer_vector(["system", "response"])
print(vector)
