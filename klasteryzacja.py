
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


dane = pd.read_csv('Desktop/jewellery.csv')


dane.head()


dane.isnull().sum().sum()



# badamy tylko dane o dochodzie i wieku
dane=dane.iloc[:,[1,3]]



sns.set_theme(style="ticks")
sns.pairplot(dane)




# wykresy wskazują, że odpowiednia liczba klastrow to 4 lub 5

# w celu wyznaczenia odpowiedniej liczby klastrow wspomozemy sie WCSS - Within-Cluster Sum-of-Squared

wcss = []
for i in range(1,10):  # wybieramy liczbe klastrow od 1 do 10
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(dane)
  wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(wcss, columns=['wcss']) # dodawanie listy wcss do dataframe
wcss = wcss.reset_index()
wcss = wcss.rename(columns={'index':'clusters'}) # zmiana nazwy kolumny
wcss['clusters']+=1
print(wcss)
px.line(wcss, x='clusters', y='wcss', width=950, height=500, title='Within-Cluster-Sum of Squared Errors (WCSS)',
        template='plotly_dark')



# wyrazne punkty zalamania wystepuja w liczbie klastrow - dwoch, trzech i czterech. Przy liczbie cztery, istnieje ostatnie wyrazne zalamanie wykresu, zatem to jest prawdopodobnie optymalna liczba klastrow



kmeans = KMeans(n_clusters=4)
kmeans.fit(dane)

y_kmeans = kmeans.predict(dane)
dane['y_kmeans'] = y_kmeans

px.scatter(dane, 'Income', 'Savings', 'y_kmeans', width=950, height=500, title='Algorytm K-średnich - 4 klastry', 
           template='plotly_dark')



# Algorytm niepoprawnie rozdzielił klastry, poniewaz poprzez brak standaryzacji, zmienna Income jest zmienną dominującą, a wykorzystywaną metryką jest euklidesowa.
# Rozwiązaniem jest poleganie na odległości Malahanobisa lub standaryzacja danych


dane.drop("y_kmeans", axis = 1, inplace= True)



list(dane)



scaler = StandardScaler()
dane[list(dane)] = scaler.fit_transform(dane[list(dane)])


wcss = []
for i in range(1,10): 
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(dane)
  wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(wcss, columns=['wcss'])
wcss = wcss.reset_index()
wcss = wcss.rename(columns={'index':'clusters'})
wcss['clusters']+=1
print(wcss)
px.line(wcss, x='clusters', y='wcss', width=950, height=500, title='Within-Cluster-Sum of Squared Errors (WCSS)',
        template='plotly_dark')



kmeans = KMeans(n_clusters=4)
kmeans.fit(dane)

y_kmeans = kmeans.predict(dane)
dane['y_kmeans'] = y_kmeans

px.scatter(dane, 'Income', 'Savings', 'y_kmeans', width=950, height=500, title='Algorytm K-średnich - 4 klastry', 
           template='plotly_dark')


# klasteryzacja poprawna


# GRUPOWANIE HIERARCHICZNE

cluster = AgglomerativeClustering(n_clusters=4)
cluster.fit_predict(dane)


df = pd.DataFrame(dane, columns=['Income', 'Savings'])
df['cluster'] = cluster.labels_

fig = px.scatter(df, 'Income', 'Savings', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne', color_continuous_midpoint=0.6)
fig.update_traces(marker_size=12)
fig.show()


cluster_euclidean = AgglomerativeClustering(n_clusters=4)
cluster_euclidean.fit_predict(dane)
df_euclidean = pd.DataFrame(dane, columns=['Income', 'Savings'])
df_euclidean['cluster'] = cluster_euclidean.labels_
fig = px.scatter(df_euclidean, 'Income', 'Savings', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka euklidesowa', color_continuous_midpoint=0.6)
fig.show()



# metryka manhattan
cluster_manhattan = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='complete')
cluster_manhattan.fit_predict(dane)
df_manhattan = pd.DataFrame(dane, columns=['Income', 'Savings'])
df_manhattan['cluster'] = cluster_manhattan.labels_
fig = px.scatter(df_manhattan, 'Income', 'Savings', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka Manhattan', color_continuous_midpoint=0.6)
fig.show()



### DBSCAN

from sklearn.cluster import DBSCAN

cluster = DBSCAN()
cluster.fit(dane)

cluster.labels_[:10]

dane['cluster'] = cluster.labels_ # dane do ramki
dane.cluster.value_counts() # liczba punktow w kazdym klastrze

# wizulizacja
px.scatter(dane, 'Income', 'Savings', 'cluster', width=950, height=500, 
           template='plotly_dark', color_continuous_midpoint=0)


