import argparse, ctypes, math, time
from pyprnt import prnt
bm25_lib = ctypes.cdll.LoadLibrary('./libbm25.so')

class BM25(object):
    def __init__(self, path, k1=1.5, b=0.75, epsilon=0.25):
        # setting types for ctypes
        bm25_lib.BM25_new.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        bm25_lib.BM25_new.restype = ctypes.c_void_p
        bm25_lib.BM25_load_from_file.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
        bm25_lib.BM25_load_from_file.restype = ctypes.c_void_p
        bm25_lib.BM25_build_tf_df.argtypes = [ctypes.c_void_p, ctypes.c_int]
        bm25_lib.BM25_build_tf_df.restype = ctypes.c_void_p
        bm25_lib.BM25_get_vocab_size.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_vocab_size.restypes = ctypes.c_int
        bm25_lib.BM25_get_vocab_lens.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_vocab_lens.restype = ctypes.POINTER(ctypes.c_int)
        bm25_lib.BM25_get_total_docs.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_total_docs.restype = ctypes.c_int
        bm25_lib.BM25_get_docs_len.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_docs_len.restype = ctypes.POINTER(ctypes.c_int)
        bm25_lib.BM25_get_vocab_list.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_vocab_list.restype = ctypes.POINTER(ctypes.c_char_p)
        bm25_lib.BM25_get_df.argtypes = [ctypes.c_void_p, ctypes.c_int]
        bm25_lib.BM25_get_df.restype = ctypes.c_int
        bm25_lib.BM25_get_df_list.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_df_list.restype = ctypes.POINTER(ctypes.c_int)
        bm25_lib.BM25_get_tf.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        bm25_lib.BM25_get_tf.restype = ctypes.c_int
        bm25_lib.BM25_get_tf_list.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_tf_list.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        bm25_lib.BM25_get_idf_list.argtypes = [ctypes.c_void_p]
        bm25_lib.BM25_get_idf_list.restype = ctypes.POINTER(ctypes.c_float)
        bm25_lib.BM25_get_score.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        bm25_lib.BM25_get_score.restype = ctypes.c_float
        bm25_lib.BM25_get_scores.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        bm25_lib.BM25_get_scores.restype = ctypes.POINTER(ctypes.c_float)

        # init class pointer and load files
        self.obj = bm25_lib.BM25_new(k1, b, epsilon)
        bm25_lib.BM25_load_from_file(self.obj, path.encode())

        #
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # internal values
        self.vocab_size = 0
        self.id_to_vocab = []
        self.vocab_to_id = {}
        self.total_docs = 0
        self.docs_len = []
        self.df = []
        self.idf = []
        self.tf = []
        self.average_dl = 0
        self.average_idf = 0

    def build(self, thread_num):
        bm25_lib.BM25_build_tf_df(self.obj, thread_num)
        self.vocab_size = bm25_lib.BM25_get_vocab_size(self.obj)
        self.total_docs = bm25_lib.BM25_get_total_docs(self.obj)
        vocab_lens_p = bm25_lib.BM25_get_vocab_lens(self.obj)
        vocab_lens = [vocab_lens_p[i] for i in range(self.vocab_size)]
        vocab_list_p = bm25_lib.BM25_get_vocab_list(self.obj)
        self.id_to_vocab = [vocab_list_p[i][:vocab_lens[i]].decode() for i in range(self.vocab_size)]
        self.vocab_to_id = { self.id_to_vocab[i] : i for i in range(self.vocab_size) }
        df_list = bm25_lib.BM25_get_df_list(self.obj)
        self.df = [df_list[i] for i in range(self.vocab_size)]
        tf_list = bm25_lib.BM25_get_tf_list(self.obj)
        docs_len_list = bm25_lib.BM25_get_docs_len(self.obj)
        self.docs_len = [docs_len_list[i] for i in range(self.total_docs)]
        idf_list = bm25_lib.BM25_get_idf_list(self.obj)
        self.idf = [idf_list[i] for i in range(self.vocab_size)]

    def get_score(self, query, did):
        if not isinstance(query, list):
            raise Exception('query argument must be list of <str> or <int>, but %s is given.' % (type(query)))
        if isinstance(query[0], str):
            query = [self.vocab_to_id[q] for q in query]
        elif isinstance(query[0], int):
            pass
        else:
            raise Exception('query argument must be list of <str> or <int>, but %s is given.' % (type(query)))
        query_arr = (ctypes.c_int * len(query))(*query)
        score = bm25_lib.BM25_get_score(self.obj, query_arr, len(query), did)
        return score

    def get_scores(self, query):
        scores = []
        for did in range(self.total_docs):
            scores.append(self.get_score(query, did))
        return scores

def mean(data):
    return sum(data) / len(data)

def var(data):
    return mean([d ** 2 for d in data]) - mean(data) ** 2

def std(data):
    return var(data) ** 0.5

parser = argparse.ArgumentParser()
parser.add_argument('-nt', '--num_threads', required=True, type=int)
parser.add_argument('-ni', '--num_iter', required=True, type=int)
args = parser.parse_args()

prnt({
    'num_threads': args.num_threads,
    'num_iter': args.num_iter
})

times = []
for it in range(1, args.num_iter + 1):
    start = time.time()
    print('Loading ... ', end='', flush=True)
    bm25 = BM25('data/rob04_docs.txt')
    end = time.time()
    print('Fin! (%d sec)' % (end - start), flush=True)

    start = time.time()
    print('[%3d/%3d] Building BM25 ... ' % (it, args.num_iter), end='', flush=True)
    bm25.build(args.num_threads)
    end = time.time()
    print('Fin! (%d sec)' % (end - start), flush=True)
    times.append(end - start)

print(times)
prnt({
    'Mean': mean(times),
    'Var': var(times),
    'Std': std(times)
})
