#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <thread>

class BM25 {
public:
    std::vector<char> buffer;
    std::map<std::string, int> vocab_to_idx;
    std::vector<std::vector<int>> docs_tok;
    std::vector<std::vector<int>> docs_tok_uni;
    pthread_mutex_t lock_vocab, lock_docs;
    std::vector<int> split_points;
    int total_vocabs = 0;
    int total_docs = 0;
    int *docs_len;
    int *vocab_lens;
    char **vocab_list;
    int **tf;
    int *df;
    float *idf;
    float average_dl = 0.0;
    float average_idf = 0.0;
    float PARAM_K1 = 1.5;
    float PARAM_B = 0.75;
    float EPSILON = 0.25;
    float *scores;

    BM25() {
        PARAM_K1 = 1.5;
        PARAM_B = 0.75;
        EPSILON = 0.25;
    }

    BM25(float param_k1, float param_b, float epsilon) {
        PARAM_K1 = param_k1;
        PARAM_B = param_b;
        EPSILON = epsilon;
    }

    void load_from_file(const char *path) {
        int ch;
        bool is_last_newline = false;
        FILE *fp = fopen(path, "r");
        while (ch = fgetc(fp)) {
            if (ch == EOF) break;
            if (ch == '\n') {
                is_last_newline = true;
            } else if (is_last_newline) {
                is_last_newline = false;
                split_points.push_back(buffer.size());
            } else {
                is_last_newline = false;
            }
            buffer.push_back(ch);
        }
        split_points.push_back(buffer.size());
        fclose(fp);
    }

    void build_tf_df_multi(int thread_num, int tid, int range_begin, int range_end) {
        // process for each thread
        int total_vocabs_local = 0;
        char vocab[100];
        char ch;
        int pos = 0;
        std::vector<std::vector<int>> docs_tok_local, docs_tok_uni_local;
        std::map<std::string, int> vocab_to_idx_local;
        std::vector<int> doc_tok, doc_tok_uni;
        for (int i = range_begin; i < range_end; i++) {
            ch = buffer[i];
            if (ch == EOF) break;
            if (ch == ' ' || ch == '\n') {
                if (pos == 0) continue;
                std::string s(vocab, pos);
                if (vocab_to_idx_local.find(s) == vocab_to_idx_local.end()) {
                    vocab_to_idx_local[s] = total_vocabs_local;
                    total_vocabs_local++;
                    doc_tok_uni.push_back(vocab_to_idx_local[s]);
                }
                doc_tok.push_back(vocab_to_idx_local[s]);
                pos = 0;
                if (ch == '\n') {
                    docs_tok_local.push_back(doc_tok);
                    docs_tok_uni_local.push_back(doc_tok_uni);
                    doc_tok = std::vector<int>();
                    doc_tok_uni = std::vector<int>();
                }
            } else {
                vocab[pos] = ch;
                pos++;
            }
        }
        // make convert table
        int *convert = new int[total_vocabs_local];
        for (auto it = vocab_to_idx_local.begin(); it != vocab_to_idx_local.end(); it++) {
            pthread_mutex_lock(&lock_vocab);
            if (vocab_to_idx.find(it->first) != vocab_to_idx.end()) {
                convert[it->second] = vocab_to_idx[it->first];
            } else {
                vocab_to_idx[it->first] = total_vocabs;
                convert[it->second] = total_vocabs;
                total_vocabs++;
            }
            pthread_mutex_unlock(&lock_vocab);
        }
        // convert and merge docs
        int docs_cnt = docs_tok_local.size();
        for (int did = 0; did < docs_cnt; did++) {
            // convert doc
            int doc_len = docs_tok_local[did].size();
            for (int i = 0; i < doc_len; i++) {
                docs_tok_local[did][i] = convert[docs_tok_local[did][i]];
            }
            int doc_uni_len = docs_tok_uni_local[did].size();
            for (int i = 0; i < doc_uni_len; i++) {
                docs_tok_uni_local[did][i] = convert[docs_tok_uni_local[did][i]];
            }
            // merge doc
            pthread_mutex_lock(&lock_docs);
            docs_tok.push_back(docs_tok_local[did]);
            docs_tok_uni.push_back(docs_tok_uni_local[did]);
            pthread_mutex_unlock(&lock_docs);
        }
    }

    void build_tf_df(int thread_num) {
        // initialize variables
        total_vocabs = 0;
        pthread_mutex_init(&lock_vocab, NULL);
        pthread_mutex_init(&lock_docs, NULL);
        // split ranges
        int *range_begin = new int[thread_num];
        int *range_end = new int[thread_num];
        int split_cnt = split_points.size();
        int split_size = (split_cnt + thread_num - 1) / thread_num;
        for (int i = 0; i < thread_num; i++) {
            if (i == 0)
                range_begin[i] = 0;
            else
                range_begin[i] = split_points[split_size * i - 1];
            if (i == thread_num - 1)
                range_end[i] = split_points[split_cnt - 1];
            else
                range_end[i] = split_points[split_size * (i + 1) - 1];
        }
        // do multi-threading
        std::vector<std::thread> workers;
        for (int i = 0; i < thread_num; i++) {
            workers.push_back(std::thread(&BM25::build_tf_df_multi, this, thread_num, i, range_begin[i], range_end[i]));
        }
        for (int i = 0; i < thread_num; i++) {
            workers[i].join();
        }
        // calculate total_docs, total_vocabs, tf, df, docs_len
        total_docs = docs_tok.size();
        total_vocabs = vocab_to_idx.size();
        df = new int[total_vocabs];
        tf = new int*[total_docs];
        docs_len = new int[total_docs];
        for (int i = 0; i < total_vocabs; i++)
            df[i] = 0;
        for (int did = 0; did < total_docs; did++) {
            int docs_len_uni = docs_tok_uni[did].size();
            docs_len[did] = docs_tok[did].size();
            tf[did] = new int[total_vocabs];
            for (int i = 0; i < docs_len_uni; i++) {
                int cnt = 0;
                for (int j = 0; j < docs_len[did]; j++) {
                    if (docs_tok[did][j] == docs_tok_uni[did][i]) {
                        cnt++;
                    }
                }
                tf[did][docs_tok_uni[did][i]] = cnt;
                if (cnt > 0) {
                    df[docs_tok_uni[did][i]]++;
                }
            }
        }
        // calculate average_dl, average_idf, idf
        average_dl = 0.0;
        for (int did = 0; did < total_docs; did++) {
            average_dl += docs_len[did];
        }
        average_dl /= total_docs;
        average_idf = 0.0;
        idf = new float[total_vocabs];
        std::vector<int> neg_idf_idx;
        for (int i = 0; i < total_vocabs; i++) {
            idf[i] = log(total_docs - df[i] + 0.5) - log(df[i] + 0.5);
            average_idf += idf[i];
            if (idf[i] < 0.0) {
                neg_idf_idx.push_back(i);
            }
        }
        average_idf /= total_vocabs;
        float eps = EPSILON * average_idf;
        for (int i = 0; i < neg_idf_idx.size(); i++) {
            idf[neg_idf_idx[i]] = eps;
        }
        // build variables for ctypes
        vocab_lens = new int[total_vocabs];
        vocab_list = new char*[total_vocabs];
        for (auto it = vocab_to_idx.begin(); it != vocab_to_idx.end(); it++) {
            int id = it->second;
            std::string word = it->first;
            vocab_lens[id] = word.size();
            vocab_list[id] = new char[word.size() + 1];
            for (int i = 0; i < word.size(); i++) {
                vocab_list[id][i] = word[i];
            }
            vocab_list[id][word.size()] = 0;
        }
    }

    float get_score(int *query, int query_len, int did) {
        float numerator_const = PARAM_K1 + 1;
        float denominator_const = PARAM_K1 * (1.0 - PARAM_B + (PARAM_B * docs_len[did] / average_dl));
        float score = 0.0;
        for (int i = 0; i < query_len; i++) {
            score += (idf[query[i]] * tf[did][query[i]] * numerator_const) / (tf[did][query[i]] + denominator_const);
        }
        return score;
    }

    float* get_scores(int *query, int query_len) {
        scores = new float[total_docs];
        for (int did = 0; did < total_docs; did++) {
            scores[did] = get_score(query, query_len, did);
        }
        return scores;
    }
};

extern "C" {
    BM25* BM25_new(float param_k1, float param_b, float epsilon) {
        return new BM25(param_k1, param_b, epsilon);
    }

    void BM25_load_from_file(BM25* bm25, const char *path) {
        bm25->load_from_file(path);
    }

    void BM25_build_tf_df(BM25* bm25, int thread_num) {
        bm25->build_tf_df(thread_num);
    }

    int BM25_get_vocab_size(BM25* bm25) {
        return bm25->total_vocabs;
    }

    int* BM25_get_vocab_lens(BM25* bm25) {
        return bm25->vocab_lens;
    }

    int BM25_get_total_docs(BM25* bm25) {
        return bm25->total_docs;
    }

    int* BM25_get_docs_len(BM25* bm25) {
        return bm25->docs_len;
    }

    char** BM25_get_vocab_list(BM25* bm25) {
        return bm25->vocab_list;
    }

    int BM25_get_df(BM25* bm25, int idx) {
        return bm25->df[idx];
    }

    int* BM25_get_df_list(BM25* bm25) {
        return bm25->df;
    }

    int BM25_get_tf(BM25* bm25, int did, int idx) {
        return bm25->tf[did][idx];
    }

    int** BM25_get_tf_list(BM25* bm25) {
        return bm25->tf;
    }

    float* BM25_get_idf_list(BM25* bm25) {
        return bm25->idf;
    }

    float BM25_get_score(BM25* bm25, int *query, int query_len, int did) {
        return bm25->get_score(query, query_len, did);
    }

    float* BM25_get_scores(BM25* bm25, int *query, int query_len) {
        return bm25->get_scores(query, query_len);
    }
}

/*
int main() {
    BM25 bm25;
    bm25.load_from_file("test.txt");
    bm25.build_tf_df(4);
    for (auto it = bm25.vocab_to_idx.begin(); it != bm25.vocab_to_idx.end(); it++) {
        int id = it->second;
        printf("%s: %d\n", it->first.c_str(), bm25.DF[id]);
    }
    return 0;
}
*/
