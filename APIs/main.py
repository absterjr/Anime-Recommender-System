import re
import json
import pandas as pd
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
templates = Jinja2Templates(directory="templates2/")

api.add_middleware(
        CORSMiddleware,
        # allow_origins=["*"],
        allow_origin_regex='https?://.*',
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def content_rec(name):



    anime = pd.read_csv('anime.csv')
 #   print(1)
    rating = pd.read_csv('rating.csv', sep=',')
#    print(2)
    df = pd.merge(anime, rating, on='anime_id')
 #   print(3)
    df.rename(columns={'rating_x': 'avg_rating', 'rating_y': 'user_rating'}, inplace=True)
 #   print(4)
    df['anime_id'] = df.anime_id.astype('object')
 #   print(5)
    df['user_id'] = df.user_id.astype('object')
 #   print(6)

    data = anime.copy()
 #   print(7)
    data['describe'] = data['genre'] + data['type'] + data['episodes']
 #   print(8)
    data['describe'].fillna(' ')
 #   print(9)
    data.drop_duplicates(subset=['name'], inplace=True)
 #   print(10)
    data.reset_index(drop=True, inplace=True)
  #  print(11)
    print(data.shape)
 #   print(12)
    tf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
  #  print(13)
  #  print(tf)
  #  print(data['describe'])
    tf_matrix = tf.fit_transform(data['describe'].values.astype('U'))
   # print(14)
  #  print(tf_matrix.shape)
  #  print(15)
    simil = linear_kernel(tf_matrix, tf_matrix)
   # print(simil.shape)
    id = data['anime_id'].values
    simil = pd.DataFrame(simil, index=id, columns=id)
    simil.columns = data['name']
    simil['anime_name'] = data['name'].values
    idx = pd.DataFrame(pd.np.empty((0, 2)))
    idx = simil[simil['anime_name']==name]
    idx = idx.drop('anime_name', axis=1).T
   # print(idx)
    idx.columns = ['similarity']
    idx = idx.sort_values(by='similarity', ascending=False)
    return idx.head(5)

@api.get("/recommendations/",response_class=HTMLResponse)
async def recommendations(request:Request):
   # if anime_name:
   #     recommendations = content_rec(anime_name, n_recommendations)
   #     recommendations = recommendations.to_json()
   #    infoFromJson = json.loads(recommendations)
   #    #return JSONResponse(json.loads(recommendations))
   # return json2html.convert(json=infoFromJson)
    result = "Anime:  "
    return templates.TemplateResponse('form2.html', context={'request': request, 'result': result})
    return JSONResponse(content={"Error": "The anime name is missing"}, status_code=400)

@api.post("/recommendations/",response_class=HTMLResponse)
def recommendations(request: Request, name: str = Form(...)):
    print(name)
    if name:
        recommendations = content_rec(name)
        recommendations = recommendations.to_json()
        infoFromJson = json.loads(recommendations)

        result = json2html.convert(json=infoFromJson)

    #return templates.TemplateResponse('form2.html', context={'request': request, 'result': result})
    return result




