import time
import faiss
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from faiss.swigfaiss import IndexFlat, IndexIVFFlat

from scipy.stats import ttest_ind
from scipy.interpolate import interp1d

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def split_join(file: list, data: dict, tag: str, exclude: tuple) -> dict:
    """
    Extract only interest tags.

    Args:
        file: Raw document.
        data: Dictionary to save the results.
        tag: Tags to keep.
        exclude: Tags to exclude.

    Returns:
        data: Precessed document.
    """
    for line in file:
        line = line.replace('\n', '').replace('\x1a', '')
        if line.startswith(tag):
            curr_doc = line[3:].replace(' ', '')
        if not line.startswith(exclude):
            data[curr_doc] += f' {line}'
    return data


def convert_queries(queries: dict,
                    embedding_model: SentenceTransformer,
                    tfidf_transformer: TfidfVectorizer) -> dict:
    """
    Vectorizes queries based on embeddings and TF-IDF.

    Args:
        queries: Dict with raw queries.
        embedding_model: Pre-trained sentece-BERT model.
        tfidf_transformer: TF-IDF vectorized.

    Returns:
        queries: Precessed queries.
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


def confussion_matrix(relevant_pred: np.array,
                      relevant_true: list,
                      collection_size: int) -> dict:
    """
    Compute true positive (tp), false positive (fp), false negative (fn) and
    true negative (tn).

    Args:
        relevant_pred: Relevant documents retrieved.
        relevant_true: Relevant documents expected.
        collection_size: Number of documents in the collection.

    Returns:
        Dict with hits, tp, fp, fn and tn.
    """
    hits = [r for r in relevant_pred if r in relevant_true]
    tp = len(hits)
    fp = len(relevant_pred) - tp
    fn = len(relevant_true) - tp
    tn = collection_size - (tp + fp + fn)
    return {'hits': hits, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def precision(tp: int, fp: int) -> float:
    """
    Compute precision metric.

    Args:
        tp: True positives.
        fp: False positives.

    Returns:
        Rate of relevant instances among the retrieved instances.
    """
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """
    Compute recall metric.

    Args:
        tp: True positives.
        fn: False negatives.

    Returns:
        Rate of relevant instances that were retrieved
    """
    return tp / (tp + fn)


def adjustment(x: np.array) -> np.array:
    """
    Adjustments applied to interpolation output.

    Args:
        x: Interpolation results.

    Returns:
        x: Adjusted interpolation values.
    """
    x[x > 1] = 1
    x[x < 0] = 0
    x[np.isnan(x) == True] = x[1]
    return x


def precision_recall(relevant_pred: np.array,
                     relevant_true: list) -> np.array:
    """
    Computes 11 standard recall levels.

    Args:
        relevant_pred: Relevant documents retrieved.
        relevant_true: Relevant documents expected.

    Returns:
        Interpolated precision values for the 11 levels of recall.
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


def p_at_n(relevant_pred: np.array,
           relevant_true: list,
           n: int) -> float:
    """
    Precision at n, to calculate the precision in the top n retrieved
    documents.

    Args:
        relevant_pred: Relevant documents retrieved.
        relevant_true: Relevant documents expected.
        n: Represents the number of ordered documents to be taken into
           account.

    Returns:
        Precision at n documents.
    """
    return len([d for d in relevant_pred[:n] if d in relevant_true]) / n


def reciprocal_rank(relevant_pred: np.array,
                    relevant_true: list,
                    threshold: int) -> float:
    """
    Computes Reciprocal rank.

    Args:
        relevant_pred: Relevant documents retrieved.
        relevant_true: Relevant documents expected.
        threshold: Threshold for ranking position.

    Returns:
        Position of the first relevant document retrieved.
    """
    for pos, doc in enumerate(relevant_pred[:threshold]):
        if doc in relevant_true:
            return pos + 1
    return 0


def mean_reciprocal_rank(rr: dict) -> float:
    """
    Computes mean reciprocal rank.

    Args:
        rr: Dictionary with all calculated metrics.

    Returns:
        Mean reciprocal rank.
    """
    mrr = 0
    for doc, results in rr.items():
        if results['reciprocal_rank'] > 0:
            mrr += 1 / results['reciprocal_rank']
    mrr /= len(rr)
    return round(mrr, 3)


def metrics(relevant_pred: np.array,
            relevant_true: list,
            collection_size: int,
            threshold: int) -> tuple:
    """
    Compute precision, recall, precision vs recall, precision at
    n documents and reciprocal rank.

    Args:
        relevant_pred: Relevant documents retrieved.
        relevant_true: Relevant documents expected.
        collection_size: Number of documents in the collection.
        threshold: Threshold for ranking position.

    Returns:
        All metrics.
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


def index_flat(vectors: np.array, add_vectors: bool = True) -> IndexFlat:
    """
    Builds the flat index.

    Args:
        vectors: Vector documents.
        add_vectors: If true indexes the vectors.

    Returns:
        idx: Index with the indexed documents.
    """
    idx = faiss.IndexFlat(vectors.shape[1], faiss.METRIC_INNER_PRODUCT)
    if add_vectors:
        idx.add(vectors)
    return idx


def index_ivfflat(vectors: np.array, nlist: int) -> IndexIVFFlat:
    """
    Builds the IVFFlat index.

    Args:
        vectors: Vector documents.
        nlist: Number of document clusters.

    Returns:
        idx: Index with the indexed documents.
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


def t_test(data: pd.DataFrame) -> None:
    """
    Calculate the T-test for the means of two independent samples of scores.
    
    Args:
        data: Dataframe with the results to be tested.

    Returns:
        None.
    """
    pre = data.Preprocessing.unique()
    var = data.variable.unique()
    for v in var:
        print('-' * 80, f'\n{v}\n')
        tmp = data[data.variable == v]
        means = [tmp[tmp.Preprocessing == p].value.values for p in pre]
        means = [(v, round(np.mean(v), 3)) for v in means]
        for i in range(len(means)):
            print(f'Average Precision - {pre[i]}: {means[i][1]}.')
        print(f'Absolute Diferrence: '
              f'{round(abs(means[0][1] - means[1][1]), 3)}.')
        print(f'p-Value: {round(ttest_ind(means[0][0], means[1][0])[1], 3)}.')


def plot_recall_levels(results: dict, title: str) -> None:
    """
    Plot the 11 standard recall levels.

    Args:
        results: Precision and recall values.
        title: Plot title.

    Returns:
        None.
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


def p_at(results: pd.DataFrame, title: str) -> None:
    """
    Violin plot for precision at n.

    Args:
        results: Values for precision at n documents.
        title: Plot title.

    Returns:
        None.
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


def mrr_plot(results: pd.DataFrame, title: str) -> None:
    """
    Plot histogram for MRR results.

    Args:
        results: Values for mean reciprocal rank.
        title: Plot title.

    Returns:
        None.
    """
    sns.countplot(x='value', hue='Type', data=results)
    plt.xlabel('Reciprocal Rank'), plt.ylabel('Count')
    plt.title(f'Reciprocal Rank ({title})')
    plt.grid(axis='y')


def search(idx: IndexFlat or IndexIVFFlat,
           k: int,
           vector_type: str,
           queries: dict,
           rel_docs: dict,
           size: int,
           threshold: int) -> dict:
    """
    Performs queries and computes metrics.

    Args:
        idx: Indexing with the indexed documents.
        k: Number of documents to be retrieved.
        vector_type: 'TF-IDF' or 'embedding'.
        queries: Processed queries.
        rel_docs: Expected documents.
        size: Number of documents in the collection.
        threshold: Threshold for ranking position.

    Returns:
        results: All metrics.
    """
    n_queries = len(queries)
    results = {}
    for qn, q in queries.items():
        print(f'{int(qn)}/{n_queries}', end='\r')
        start = time.time()
        D, I = idx.search(q[vector_type], k)
        end = time.time()
        returned_docs = I[0] + 1
        r = metrics(returned_docs, rel_docs[qn], size, threshold)
        results[qn] = {'docs_retrieved': returned_docs,
                       'hits': r[0],
                       'precision': r[1],
                       'recall': r[2],
                       'precision_recall': r[3],
                       'p@5': r[4],
                       'p@10': r[5],
                       'reciprocal_rank': r[6],
                       'query_execution_time': end - start}
    return results
