# O que é preciso ser feito

O objetivo do trabalho prático é implementar e avaliar modelos de recuperação de informação usando a
biblioteca de busca por similaridade FAISS (Facebook AI Similarity Search). A implementação deve ser feita
de acordo com as seguintes instruções:

- Para avaliação do resultado, use a coleção de referência CFC (Cystic Fibrosis). Para o gabarito de
resultado das consultas dessa coleção, desconsidere o score de relevância, apenas considere todos
os documentos no resultado como relevantes. Para indexação dos documentos da coleção CFC, crie
um único campo, concatenando os textos contidos nos atributos AU, TI, SO, MJ, MN e AB/EX. Faça o
pré-processamento do texto, se necessário;
- Para vetorização dos documentos (e consultas), use duas codificações diferentes de embeddings:
TF-IDF e Sentence Transformers;
- Para indexação dos documentos, experimente um ou mais índices disponíveis no FAISS;
- Para efetuar as consultas e obter seus resultados, use a similaridade do cosseno;
- Para avaliação do resultado, crie um módulo para retornar as métricas seguintes. O módulo deve
receber como entrada a identificação das consultas com as respectivas identificações dos
documentos de seu resultado, informados pelo gabarito da coleção de referência e pelo seu
algoritmo, ordenados por relevância, e retornar os valores para as métricas Precisão e Revocação. O
módulo deve gerar a tabela de Precisão x Revocação para 11 níveis de revocação, para que seja
montado um único gráfico com os valores médios entre todas as consultas da coleção. Além disso,
calcule também os valores para as métricas P@5 e P@10 médios, MRR(Q) considerando o threshold
S h = 5 e trace o histograma da precisão-R para as 20 primeiras consultas;
- Avalie cada codificação de embeddings separadamente, apresente os seus respectivos gráficos de
Precisão x Revocação e os resultados das demais métricas descritas acima.

# Organização do repo

O repositório está organizado da seguinte forma:

- cfc: Contém os arquivos brutos da coleção
- notebooks: Contém os códigos e as execuções dos experimentos
- outputs: Contém a coleção e as consultas processadas
