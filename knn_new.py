import os
import random

import pandas as pd
from connect import conn
from fuzzywuzzy import fuzz
from scipy.sparse import csr_matrix
from flask import jsonify
df_feature = pd.read_sql_query('SELECT * FROM features', conn)

df_product = pd.read_sql_query('SELECT * FROM products', conn)
df_product = df_product.fillna(0)
df_product_feature = pd.read_sql_query('SELECT * FROM feature_detail', conn)
df_product_feature = df_product_feature.join(df_feature.set_index('feature_id'), on='feature_id')

# pivot ratings into movie features
df_product_features = df_product_feature.pivot(
    index='product_id',
    columns='feature_type_id',
    values='point'
).fillna(0)
products_to_idx = {
    product: i for i, product in
    enumerate(list(df_product.set_index('id').loc[df_product_features.index].name))
}

def fuzzy_matching(mapper, fav_product, verbose=True):
    """
    return the closest match via fuzzy ratio.

    Parameters
    ----------
    mapper: dict, map movie title name to index of the movie in data

    fav_product: str, name of user input movie

    verbose: bool, print log if True

    Return
    ------
    index of the closest match
    """
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_product.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    # if verbose:
    #     print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]


from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)


def save_model():
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)


# if __name__ == "__main__":
#     save_csv()
def make_recommendation(fav_product,model_knn=model_knn,
    data=csr_matrix(df_product_features.values),

    mapper=products_to_idx,
    n_recommendations=10):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model

    data: product-feature matrix

    mapper: dict, map product title name to index of the movie in data

    fav_product: str, name of user input product

    n_recommendations: int, top n recommendations

    Return
    ------
    list of top n similar product recommendations
    """
    # fit
    model_knn.fit(data)
    # get input movie index
    #print('You have input product:', fav_product)
    idx = fuzzy_matching(mapper, fav_product, verbose=True)
    if idx is None:
        return []
    #print('Recommendation system start to make inference')
    #print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations + 1)

    raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[
                     :0:-1]

    # get reverse mapper
    #print(raw_recommends)
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    #print('Recommendations for {}:'.format(fav_product))
    filter = []
    for i, (idx, dist) in enumerate(raw_recommends):
        #print('{0}: {1}, with distance of {2}'.format(i + 1, reverse_mapper[idx], dist))
        filter.append(reverse_mapper[idx])


    newproduct = pd.read_sql_query("""SELECT * FROM products where name IN %s""", conn,params=(tuple(filter),))

    return newproduct.reset_index().to_json(orient='records')

# In[26]:

# my_favorite = 'MRE92LL/A'
#
# print(make_recommendation(my_favorite))
