import os
import random

import pandas as pd
import psycopg2
from datetime import datetime

from connect import conn

data_path = 'ml-20m/'
ftyoe_filename = 'featureType.csv'
feature_filename = 'feature.csv'
products_filename = 'products.csv'

df_feature = pd.read_csv(
    os.path.join(data_path, feature_filename),

    usecols=['id', 'spec', 'type', 'point'],
    dtype={'id': 'int32', 'spec': 'str', 'type': 'str', 'point': 'int32'})

df_type = pd.read_csv(
    os.path.join(data_path, ftyoe_filename),

    usecols=['id', 'name'],
    dtype={'id': 'str', 'name': 'str'})

df_product = pd.read_csv(
    os.path.join(data_path, products_filename),
    usecols=['Model', 'Category', 'SKU', 'Color', 'Storage', 'CPU', 'RAM', 'Graphics', 'Resolution', 'Price'],
    dtype={'Model': 'str', 'Category': 'int32', 'SKU': 'int32', 'Color': 'str', 'Storage': 'int32', 'CPU': 'int32',
           'RAM': 'int32', 'Graphics': 'str', 'Resolution': 'int32', 'Price': 'str'})
df_product = df_product.fillna(0)

from scipy.sparse import csr_matrix

df_product_feature = pd.read_csv(os.path.join(data_path, 'product-feature.csv'))

df_product_feature = df_product_feature.join(df_feature.set_index('id'), on='feature_id')

# pivot ratings into movie features
df_product_features = df_product_feature.pivot(
    index='product_id',
    columns='type',
    values='point'
).fillna(0)
#print(df_product_features)


def insert():
    sql = """INSERT INTO products(id,create_date,price,name,category_id,remain)
                 VALUES(%s,%s,%s,%s,%s,%s);"""
    # print(df_product[:1])
    feature_sql = """INSERT INTO feature_detail(product_id,feature_id)
                 VALUES(%s,%s);"""
    cur = None
    vendor_id = None
    try:

        cur = conn.cursor()
        # execute the INSERT statement
        for product in df_product.values:
            # print((product[2], datetime.now(), float(product[9].replace(',', '')), product[0], product[1],
            # random.randint(1, 100)))
            if product[3] != 0:
                cur.execute(feature_sql, (product[2], product[3],))
            if product[4] != 0: cur.execute(feature_sql, (product[2], product[4],))
            if product[5] != 0: cur.execute(feature_sql, (product[2], product[5],))
            if product[6] != 0: cur.execute(feature_sql, (product[2], product[6],))
            if product[7] != 0: cur.execute(feature_sql, (product[2], product[7],))
            if product[8] != 0: cur.execute(feature_sql, (product[2], product[8],))

        conn.commit()

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def save_csv():
    df = pd.DataFrame(columns=['product_id', 'feature_id'])

    for product in df_product.values:
        if product[3] != 0:
            df = df.append({'product_id': product[2], 'feature_id': product[3]}, ignore_index=True)
        if product[4] != 0: df = df.append({'product_id': product[2], 'feature_id': product[4]}, ignore_index=True)
        if product[5] != 0: df = df.append({'product_id': product[2], 'feature_id': product[5]}, ignore_index=True)
        if product[6] != 0: df = df.append({'product_id': product[2], 'feature_id': product[6]}, ignore_index=True)
        if product[7] != 0: df = df.append({'product_id': product[2], 'feature_id': product[7]}, ignore_index=True)
        if product[8] != 0: df = df.append({'product_id': product[2], 'feature_id': product[8]}, ignore_index=True)
    print(df)
    df.to_csv('product-feature.csv', index=False)

from fuzzywuzzy import fuzz


# In[24]:


def fuzzy_matching(mapper, fav_movie, verbose=True):
    """
    return the closest match via fuzzy ratio.

    Parameters
    ----------
    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
def save_model():
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

# if __name__ == "__main__":
#     save_csv()
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: movie-user matrix

    mapper: dict, map movie title name to index of the movie in data

    fav_movie: str, name of user input movie

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar movie recommendations
    """
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)

    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)

    raw_recommends =         sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

print(df_product_features)
# In[26]:
movie_to_idx = {
    movie: i for i, movie in
    enumerate(list(df_product.set_index('SKU').loc[df_product_features.index].Model))
}

my_favorite = 'MPXT2LL/A'

make_recommendation(
    model_knn=model_knn,
    data=csr_matrix(df_product_features.values),
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)
