import re
import json
import pandas as pd
import string
from apyori import apriori
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import streamlit as st
from json2html import *
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

api = FastAPI()
templates = Jinja2Templates(directory="templates/")
api.add_middleware(
        CORSMiddleware,
        # allow_origins=["*"],
        allow_origin_regex='https?://.*',
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_ar_type(type):
    anime = pd.read_csv('anime.csv')
    rating = pd.read_csv('rating.csv', sep=',')
    type_ul = []
    grouped = rating.groupby("user_id")
    set_type = set(anime[anime["type"] == type]["anime_id"].values)
    print(type + ' :', len(set_type))

    for i in rating['user_id'].unique():
        g = grouped.get_group(i)
        r = g[g['rating'] >= 6]
        set_trans = set(r['anime_id'].values)
        anime_type = list(set_type.intersection(set_trans))
        if len(anime_type) > 1:
            type_ul.append(anime_type)

    association_rules = apriori(type_ul, min_support=0.15, min_confidence=0.4, min_lift=1)
    association_results = list(association_rules)

    Result = pd.DataFrame(columns=['Previously Watched', 'Recommended'])
    for item in association_results:
        pair = item[2]
        for i in pair:
            items = str([x for x in i[0]])
            if i[3] != 1:
                Result = Result.append({'Previously Watched': str(
                    [anime[anime['anime_id'] == x].reset_index().loc[0, 'name'] for x in i[0]]), 'Recommended': str(
                    [anime[anime['anime_id'] == x].reset_index().loc[0, 'name'] for x in i[1]])}, ignore_index=True)
    Result_ar = Result.sort_values(by='Previously Watched', ascending=False)
    Result_ar = Result_ar.reset_index(drop=True)
    return Result_ar.head(5)








@api.get("/api/v1/recommendations/",response_class=HTMLResponse)
async def recommendations(request: Request):
    #if type:
       # recommendations = get_ar_type(type)
       # recommendations = recommendations.to_json()
      #  infoFromJson = json.loads(recommendations)
        #print(json2html.convert(json=infoFromJson))
        #return json2html.convert(json=infoFromJson)
        #return JSONResponse(json.loads(recommendations))
    result = "Anime:  "
    return templates.TemplateResponse('form.html', context={'request': request, 'result': result})

    return JSONResponse(content={"Error": "The anime name is missing"}, status_code=400)

@api.post("/api/v1/recommendations/",response_class=HTMLResponse)
def recommendations(request: Request, name: str = Form(...)):
    if name:
        recommendations = get_ar_type(name)
        recommendations = recommendations.to_json()
        infoFromJson = json.loads(recommendations)
        result = json2html.convert(json=infoFromJson)

    #return templates.TemplateResponse('form.html', context={'request': request, 'result': result})
    return result
