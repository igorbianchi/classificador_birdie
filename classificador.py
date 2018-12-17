import pandas
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# pontuações e símbolos de um carácter desnecessárias nos títulos
tradutor = {43:None, 167:None,40:None, 41:None, 124:" ", 42:None, 33:None, 47:" ", 34:None, 39:None}

# um data-frame como base para aplicação do k-means e outro como base para exportação final
df = pandas.read_csv('smartphone.csv', sep='\t')
esquerda = pandas.read_csv('smartphone.csv', sep='\t')

for index, linha in df.iterrows():
    titulo = df['TITLE'][index]
    # remoção de pontuação com mais de um caracter e conjunções
    titulo = titulo.replace(" e ", " ")
    titulo = titulo.translate(tradutor)
    titulo = titulo.replace(", ", " ")
    titulo = titulo.replace(". ", " ")
    titulo = titulo.replace(" - ", " ")
    titulo = titulo.replace(" com ", " ")
    titulo = titulo.replace(" de ", " ")
    titulo = titulo.replace(" para ", " ")
    # remoção de palavras com apenas 1 caracter
    titulo = ' '.join([w for w in titulo.split() if len(w)>1])
    df['TITLE'][index] = titulo

# transformação das expressões em valores numéricos
vectorizer = TfidfVectorizer(min_df=10,max_df=0.8,strip_accents='unicode',lowercase=True)
X = vectorizer.fit_transform(df['TITLE'])

# aplicação do k-means
# são utilizados 3 clusters pois foi observado uma melhora nos resultados de falsos positivos
modelkmeans = KMeans(n_clusters=3)
modelkmeans.fit(X)
categorias = modelkmeans.predict(X)

# dicionário dos id's de produtos com a categoria
categoria = {'ID':esquerda['ID'],'CATEGORY':categorias}

# inserindo a coluna de categoria na tabela
direita = pandas.DataFrame(data=categoria)
output = pandas.merge(esquerda,direita,on='ID')

# o item de índice 6 é usado como base para saber se é smartphone ou não
aux = output['CATEGORY'][6]

# substituição de categoria numérica por literal
for i in range(0, len(output)):
    if (output['CATEGORY'][i] == aux):
        output['CATEGORY'][i] = "Smartphone"
    else:
        output['CATEGORY'][i] = "Não-smartphone"

# exportar para csv
output.to_csv("categorizados.csv", sep=';', encoding='utf-8', index=False)