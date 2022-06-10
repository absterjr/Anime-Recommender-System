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
         anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
         m = anime['members'].quantile(0.75)
         anime = anime[(anime['members'] >= m)]
         rating = pd.read_csv("rating.csv")
         rating.loc[rating.rating == -1, 'rating'] = np.NaN
         anime_index = pd.Series(anime.index, index=anime.name)
         joined = anime.merge(rating, how='inner', on='anime_id')
         joined = joined[['user_id', 'name', 'rating_y']]
         joined = joined[(joined['user_id'] <= 10000)]

         pivot = pd.pivot_table(joined, index='user_id', columns='name', values='rating_y')
         pivot.dropna(axis=0, how='all', inplace=True)
         pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
         pivot_norm.fillna(0, inplace=True)
         user_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index,
                                   columns=pivot_norm.index)
         if user_id not in pivot_norm.index:
          return None, None
         else:
          sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:]
          sim_score = user_sim_df.sort_values(by=user_id, ascending=False).loc[:, user_id].tolist()[1:]
          return sim_users, sim_score

def get_recommendation(user_id):
        users, score = get_similar_user(user_id)
        users, score = get_similar_user(3)
        for x, y in zip(users[:10], score[:10]):
                print("User {} with similarity of {}".format(x, y))


        # there is no information for this user
        if users is None or score is None:
                return None

        # only take 10 nearest users
        user_arr = np.array([x for x in users[:10]])
        sim_arr = np.array([x for x in score[:10]])
        predicted_rating = np.array([])
        anime = pd.read_csv("anime.csv")
        anime = anime[(anime['type'] == 'TV') | (anime['type'] == 'Movie')]
        m = anime['members'].quantile(0.75)
        anime = anime[(anime['members'] >= m)]
        rating = pd.read_csv("rating.csv")
        rating.loc[rating.rating == -1, 'rating'] = np.NaN
        anime_index = pd.Series(anime.index, index=anime.name)
        joined = anime.merge(rating, how='inner', on='anime_id')
        joined = joined[['user_id', 'name', 'rating_y']]
        joined = joined[(joined['user_id'] <= 10000)]

        pivot = pd.pivot_table(joined, index='user_id', columns='name', values='rating_y')
        pivot.dropna(axis=0, how='all', inplace=True)
        pivot_norm = pivot.apply(lambda x: x - np.nanmean(x), axis=1)
        pivot_norm.fillna(0, inplace=True)
        user_sim_df = pd.DataFrame(cosine_similarity(pivot_norm, pivot_norm), index=pivot_norm.index,
                                   columns=pivot_norm.index)
        if user_id not in pivot_norm.index:
                return None, None
        else:
                sim_users = user_sim_df.sort_values(by=user_id, ascending=False).index[1:]
                sim_score = user_sim_df.sort_values(by=user_id, ascending=False).loc[:, user_id].tolist()[1:]
                users, scores = sim_users, sim_score
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
        print(temp)
        print(anime.loc[anime_index.loc[temp.name[:10]]])
        return anime.loc[anime_index.loc[temp.name[:10]]]

@api.get("/user_recommendations/",response_class=HTMLResponse)
async def recommendations(request: Request):
   # if anime_name:
   #     recommendations = content_rec(anime_name, n_recommendations)
   #     recommendations = recommendations.to_json()
   #    infoFromJson = json.loads(recommendations)
   #    #return JSONResponse(json.loads(recommendations))
   # return json2html.convert(json=infoFromJson)
    result = "Anime:  "
    return templates.TemplateResponse('form4.html', context={'request': request, 'result': result})
    return JSONResponse(content={"Error": "The anime name is missing"}, status_code=400)

@api.post("/user_recommendations/",response_class=HTMLResponse)
def recommendations(request: Request, userid: int = Form(...)):
    print(userid)
    if userid:
        recommendations = get_recommendation(userid)
        recommendations = recommendations.to_json()
        infoFromJson = json.loads(recommendations)

        result = json2html.convert(json=infoFromJson)

    #return templates.TemplateResponse('form4.html', context={'request': request, 'result': result})
    return result
