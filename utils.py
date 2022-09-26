import faiss
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def split_join(file, data, tag, exclude):
    """Extract only interest tags."""
    for line in file:
        line = line.replace('\n', '').replace('\x1a', '')
        if line.startswith(tag):
            curr_doc = line[3:].replace(' ', '')
        if not line.startswith(exclude):
            data[curr_doc] += f' {line}'
    return data


def convert_queries(queries, embedding_model, tfidf_transformer):
    """
    Description: Vectorizes queries based on embeddings and TF-IDF.

    Args:
        queries:
        embedding_model:
        tfidf_transformer:
    """
    for qn, q in queries.items():
        # embedding
        qe = embedding_model.encode([q])
        faiss.normalize_L2(qe)

        # tf-idf
        qt = tfidf_transformer.transform([q]).toarray().astype(np.float32)

        # saving
        queries[qn] = {'original': q, 'embedding': qe, 'tf-idf': qt}

    return queries


def confussion_matrix(relevant_pred, relevant_true, collection_size):
    """
    Description: Compute true positive (tp), false positive (fp), false negative
                 (fn) and true negative (tn).

    Args:
        relevant_pred:
        relevant_true:
        collection_size:
    """
    hits = [r for r in relevant_pred if r in relevant_true]
    tp = len(hits)
    fp = len(relevant_pred) - tp
    fn = len(relevant_true) - tp
    tn = collection_size - (tp + fp + fn)
    return {'hits': hits, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def precision(tp, fp):
    """
    Description: Compute precision metric.

    Args:
        tp:
        fp:
    """
    return tp / (tp + fp)


def recall(tp, fn):
    """
    Description: Compute recall metric.

    Args:
        tp:
        fn:
    """
    return tp / (tp + fn)


def adjustment(x):
    """
    Description:

    Args:
        x:
    """
    x[x > 1] = 1
    x[x < 0] = 0
    x[np.isnan(x) == True] = x[1]
    return x


def precision_recall(relevant_pred, relevant_true):
    """
    Description: Computes 11 standard recall levels.

    Args:
        relevant_pred:
        relevant_true:
    """

    # variables to save hits, precision and recall
    hit = 0
    rel_size = len(relevant_true)
    p = np.zeros(len(relevant_pred))
    r = np.zeros(len(relevant_pred))

    # Compute hits, precsion and recall
    for idx, doc in enumerate(relevant_pred):
        if doc in relevant_true:
            hit += 1
        p[idx] = hit / (idx + 1)
        r[idx] = hit / rel_size

    # computes recall levels
    new_recall = np.arange(0, 1.1, 0.1)
    new_precision = new_recall * 0
    if np.sum(p) > 0:
        f = interp1d(r, p, fill_value="extrapolate")
        new_precision = adjustment(f(new_recall))

    return np.c_[np.array([new_recall]).T,
                 np.array([new_precision]).T]


def p_at_n(relevant_pred, relevant_true, n):
    """
    Description: Precision at n, to calculate the precision in the top n
    retrieved documents.

    Args:
        relevant_pred:
        relevant_true:
        n:
    """
    return len([d for d in relevant_pred[:n] if d in relevant_true]) / n


def reciprocal_rank(relevant_pred, relevant_true, threshold):
    """
    Description: Computes Reciprocal rank.

    Args:
        relevant_pred:
        relevant_true:
        threshold:
    """
    for pos, doc in enumerate(relevant_pred):
        if doc in relevant_true:
            return pos + 1 if pos < threshold else 0
    return 0


def mean_reciprocal_rank(rr):
    """
    Computes mean reciprocal rank.

    Args:
        rr:
    """
    mrr = 0
    for doc, results in rr.items():
        mrr += results['reciprocal_rank']
    mrr /= len(rr)
    return mrr


def metrics(relevant_pred, relevant_true, collection_size, threshold):
    """
    Description: Compute precision, recall, precision vs recall, precision at
    n,
    ...

    Args:
        relevant_pred:
        relevant_true:
        collection_size:
        threshold:
    """

    # compute the terms
    terms = confussion_matrix(relevant_pred, relevant_true, collection_size)

    # compute precision and recall
    p = precision(terms['tp'], terms['fp'])
    r = recall(terms['tp'], terms['fn'])
    p5 = p_at_n(relevant_pred, relevant_true, 5)
    p10 = p_at_n(relevant_pred, relevant_true, 10)
    p_r = precision_recall(relevant_pred, relevant_true)

    # compute reciprocal rank
    rr = reciprocal_rank(relevant_pred, relevant_true, threshold)

    return terms['hits'], p, r, p_r, p5, p10, rr


def index_flat(vectors, add_vectors=True):
    """
    Description: Builds the index.

    Args:
        vectors:
        add_vectors:
    """
    idx = faiss.IndexFlat(vectors.shape[1], faiss.METRIC_INNER_PRODUCT)
    if add_vectors:
        idx.add(vectors)
    return idx


def index_ivfflat(vectors, nlist):
    """
    Description:

    Args:
        vectors:
        nlist:
    """
    quantizer = index_flat(vectors, False)
    idx = faiss.IndexIVFFlat(
        quantizer,
        vectors.shape[1],
        nlist,
        faiss.METRIC_INNER_PRODUCT)
    idx.train(vectors)
    idx.add(vectors)
    return idx


def plot_recall_levels(results, title):
    """
    Description: Plot the 11 standard recall levels.

    Args:
        results:
        title:
    """
    colors = ['k', 'r', 'b', 'g']
    for i in range(len(results)):
        sns.lineplot(np.arange(0, 1.1, 0.1),
                     list(results.values())[i],
                     color=colors[i],
                     label=list(results.keys())[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'11 Standard Recall Levels ({title})')
    plt.grid(axis='y')


def p_at(results, title):
    """
    Description: Violin plot for precision at n.

    Args:
        results:
        title:
    """
    plt.figure(figsize=(8, 4))
    sns.violinplot(y='variable', x='value', hue='Preprocessing',
                   inner="quart", split=True, orient='h', data=results,
                   linewidth=1,
                   palette={"no preprocessing": "b", "no stopwords": ".85"})
    plt.xlabel('Precision'), plt.ylabel('')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()


def mrr_plot(results, title):
    """
    Description: Plot histogram for MRR results.

    Args:
        results:
        title:
    """
    sns.countplot(x='value', hue='Type', data=results)
    plt.xlabel('Reciprocal Rank'), plt.ylabel('Count')
    plt.title(f'Reciprocal Rank ({title})')
    plt.grid(axis='y')


def search(idx, k, vector_type, queries, rel_docs, size, threshold):
    """
    Description:

    Args:
        idx:
        k:
        vector_type:
        queries:
        rel_docs:
        size:
        threshold:
    """
    n_queries = len(queries)
    results = {}
    for qn, q in queries.items():
        print(f'{int(qn)}/{n_queries}', end='\r')
        D, I = idx.search(q[vector_type], k)
        returned_docs = I[0] + 1
        r = metrics(returned_docs, rel_docs[qn], size, threshold)
        results[qn] = {'docs_retrieved': returned_docs,
                       'hits': r[0],
                       'precision': r[1],
                       'recall': r[2],
                       'precision_recall': r[3],
                       'p@5': r[4],
                       'p@10': r[5],
                       'reciprocal_rank': r[6]}
    return results
