from gensim.models import KeyedVectors
import pdb
import numpy
from xmlrpc.server import SimpleXMLRPCServer
import sys
import time

if len(sys.argv) < 2:
    print("python3 ./go.py FILE_NAME_IN_W2V_TEXT_FORMAT")
    sys.exit(0)

fname = sys.argv[1]
started_at = time.time()

print("loading w2v {0}".format(fname))
model = KeyedVectors.load_word2vec_format(fname)
print("done in {0} sec, dim={1}".format(int(time.time() - started_at), model['word'].shape[0]))
print("Testing, similar to `king` is {0}".format(model.similar_by_word('king')))
# pdb.set_trace()

def find_weights(vocabulary_inv, num_features, miss_min, miss_max):
    unknowns = []
    res = []
    unknown_indexes = []
    print("Finding weights, vocabulary length={0}, missed randomized with {1}..{2}".format(len(vocabulary_inv), miss_min, miss_max))
    for i, w in enumrate(vocabulary_inv):
        if w in model.vocab:
            res.append(model[w][0:num_features].tolist())
        else:
            res.append(numpy.random.uniform(miss_min, miss_max, num_features).tolist())
            unknowns.append(w)
            unknown_indexes.append(i)
    print("Requested dim={0}, Missed tokens={1}".format(num_features, len(unknowns)))
    return [res, unknowns, unknown_indexes]



if __name__ == '__main__':
    server = SimpleXMLRPCServer(("localhost", 5555))
    print("Listening on port 5555...\nRPC signature: find_weights(vocabulary_inv, num_features, miss_min, miss_max)")
    server.register_function(find_weights, "find_weights")
    server.serve_forever()
