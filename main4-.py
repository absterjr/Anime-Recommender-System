import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import streamlit as st
from pydantic import BaseModel

from json2html import *
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
api = FastAPI()
templates = Jinja2Templates(directory="templates4/")
api.add_middleware(
        CORSMiddleware,
        # allow_origins=["*"],
        allow_origin_regex='https?://.*',
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
def get_similar_user(user_id):
         anime = pd.read_csv("anime.csv")
         print(1)
         anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
         print(2)
         m = anime['members'].quantile(0.75)
         print(3)
         anime = anime[(anime['members'] >= m)]
         print(4)
         rating = pd.read_csv("rating.csv")
         print(5)
         rating.loc[rating.rating == -1, 'rating'] = np.NaN
         print(6)
         anime_index = pd.Series(anime.index, index=anime.name)
         print(7)
         joined = anime.merge(rating, how='inner', on='anime_id')
         print(8)
         joined = joined[['user_id', 'name', 'rating_y']]
         print(9)
         joined = joined[(joined['user_id'] <= 10000)]
         print(10)

         pivot = pd.pivot_table(joined, index='user_id', columns='name', values='rating_y')
         print(11)
         pivot.dropna(axis=0, how='all', inplace=True)
         print(12)
         pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
         print(13)
         pivot_norm.fillna(0, inplace=True)
         print(14)
         user_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index,
                                   columns=pivot_norm.index)
         print(15)
         if user_id not in pivot_norm.index:
          return None, None

         else:
           sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:]
           sim_score = user_sim_df.sort_values(by=user_id, ascending=False).loc[:, user_id].tolist()[1:]
           return sim_users, sim_score
         print(16)

def get_recommendation(user_id):
         anime = pd.read_csv("anime.csv")
         print(17)
         anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
         print(18)
         m = anime['members'].quantile(0.75)
         anime = anime[(anime['members'] >= m)]
         rating = pd.read_csv("rating.csv")
         print(19)
         rating.loc[rating.rating == -1, 'rating'] = np.NaN
         anime_index = pd.Series(anime.index, index=anime.name)
         print(20)
         joined = anime.merge(rating, how='inner', on='anime_id')
         joined = joined[['user_id', 'name', 'rating_y']]
         joined = joined[(joined['user_id'] <= 10000)]
         print(21)

         pivot = pd.pivot_table(joined, index='user_id', columns='name', values='rating_y')
         pivot.dropna(axis=0, how='all', inplace=True)
         print(22)
         pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
         pivot_norm.fillna(0, inplace=True)
         print(23)
         user_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index,
                                   columns=pivot_norm.index)
         print(24)
         print(pivot_norm.index)
         print(user_id)
         if user_id not in pivot_norm.index:
          return None, None
         else:
          sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:]
          sim_score = user_sim_df.sort_values(by=user_id, ascending=False).loc[:, user_id].tolist()[1:]
          users, scores = sim_users, sim_score
          print(25)
# there is no information for this user
         if users is None or scores is None:
                 return None

        # only take 10 nearest users
         user_arr = np.array([x for x in users[:10]])
         sim_arr = np.array([x for x in scores[:10]])
         predicted_rating = np.array([])
         print(26)

         for anime_name in pivot_norm.columns:
                filtering = pivot_norm[anime_name].loc[user_arr] != 0.0
                temp = np.dot(pivot[anime_name].loc[user_arr[filtering]], sim_arr[filtering]) / np.sum(
                        sim_arr[filtering])
                predicted_rating = np.append(predicted_rating, temp)

        # don't recommend something that user has already rated
         temp = pd.DataFrame({'predicted': predicted_rating, 'name': pivot_norm.columns})
         filtering = (pivot_norm.loc[user_id] == 0.0)
         temp = temp.loc[filtering.values].sort_values(by='predicted', ascending=False)

        # recommend n_anime anime
         return anime.loc[anime_index.loc[temp.name[:10]]]
print(27)

@api.get("/user_recommendations/{user_id}",response_class=HTMLResponse)
async def recommendations(user_id):
    if user_id:
       recommendations = get_recommendation(user_id)
       recommendations = pd.DataFrame(data=recommendations)
       print(recommendations)
       recommendations = recommendations.to_json()

       infoFromJson = json.loads(recommendations)
       return json2html.convert(json=infoFromJson)

    return JSONResponse(content={"Error": "The anime name is missing"}, status_code=400)
