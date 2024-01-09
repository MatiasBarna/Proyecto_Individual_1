from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd
import numpy as np  
from sklearn.utils.extmath           import randomized_svd
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

columnstouse=['item_id','playtime_forever','user_id']
columnsgame=["name","price","item_id","developer","year","Indie","Action","Casual","Adventure","Strategy","Simulation","RPG","Free to Play","Early Access","Sports","Massively Multiplayer","Racing","Design &amp; Illustration","Utilities"]
df_UserItems=pd.read_parquet("df_UserItems.parquet",columns=columnstouse)
df_SteamGames=pd.read_parquet("df_SteamGames.parquet",columns=columnsgame)
df_UserReviews=pd.read_parquet("df_UserReviews.parquet")

df_SteamGames=df_SteamGames.head(7000)
df_UserItems=df_UserItems.head(7000)
df_UserReviews=df_UserReviews.head(7000)


app=FastAPI()

#http://127.0.0.1:8000

@app.get("/")
def index():
    return "Hola, bienvenid@ a mi proyecto"

@app.get('/genero/{genero}')
def PlayTimeGenre(genero:str):
    # Se filtra el DataFrame original para obtener solo las columnas relevantes
    game_columns = ['item_id', 'name', 'year', 'Indie', 'Action', 'Casual', 'Adventure', 'Strategy',
                    'Simulation', 'RPG', 'Free to Play', 'Early Access', 'Sports',
                    'Massively Multiplayer', 'Racing', 'Design &amp; Illustration', 'Utilities']

    # Se crea un nuevo DataFrame llamado df_games que contiene solo las columnas especificadas en game_columns.
    df_games = df_SteamGames[game_columns]

    # Se hace un join entre df_games y df_UserItems utilizando 'item_id' como clave
    df_joined = pd.merge(df_games, df_UserItems[['item_id', 'playtime_forever']], on='item_id', how='inner')

    # Se filtra el DataFrame resultante para el género específico
    genre_df = df_joined[df_joined[genero] == 1]

    # Se calcula la suma de horas jugadas por año y se determina el año con más horas jugadas
    total_hours_by_year = genre_df.groupby('year')['playtime_forever'].sum()
    max_year = total_hours_by_year.idxmax()

    # Se retorna el resultado en un formato específico
    result = {"Año de lanzamiento con más horas jugadas para Genero {}: {}".format(genero, max_year)}
    return result


@app.get('/usuario_por_genero/{genero}')
def UserForGenre(genero:str):
    # Se filtra el DataFrame original para obtener solo las columnas relevantes
    game_columns = ['item_id', 'name', 'year', 'Indie', 'Action', 'Casual', 'Adventure', 'Strategy',
                    'Simulation', 'RPG', 'Free to Play', 'Early Access', 'Sports',
                    'Massively Multiplayer', 'Racing', 'Design &amp; Illustration', 'Utilities']
    
    # Se crea un nuevo DataFrame llamado df_games que contiene solo las columnas especificadas en game_columns.
    df_games = df_SteamGames[game_columns]
    
    # Se hace un join entre df_games y df_UserItems utilizando 'item_id' como clave
    df_joined = pd.merge(df_games, df_UserItems[['item_id', 'playtime_forever','user_id']], on='item_id', how='inner')

    # Se filtra el DataFrame resultante para el género específico
    genre_df = df_joined[df_joined[genero] == 1]

    # Se agrupa por usuario y año, sumamos las horas jugadas y se encuentra el usuario con más horas jugadas para el género dado
    total_hours_by_user_and_year = genre_df.groupby(['user_id', 'year'])['playtime_forever'].sum()
    max_user = total_hours_by_user_and_year.groupby('user_id').sum().idxmax()


    # Se obtiene la lista de acumulación de horas jugadas por año para el usuario con más horas jugadas
    max_user_hours_by_year = total_hours_by_user_and_year.loc[max_user].reset_index()
    max_user_hours_list = [{"Año": int(row['year']), "Horas": row['playtime_forever']} for _, row in max_user_hours_by_year.iterrows()]

    # Se retorna el resultado en un formato específico
    result = {"Usuario con más horas jugadas para Género {}".format(genero): max_user, "Horas jugadas": max_user_hours_list}
    return result

@app.get('/top3_juegos_mas_recomendados/{anio}')
def UsersRecommend(año:int):
    # Se filtra el DataFrame resultante para el año específico
    reduce_df = df_UserReviews[(df_UserReviews['year'] == año) & 
                                 (df_UserReviews['recommend'] == True) & 
                                 (df_UserReviews['sentiment_analysis'] >= 1)]
    
    # Se hace un join entre df_UserReviews y df_steamgames utilizando 'item_id' como clave
    merged_reviews = pd.merge(reduce_df, df_SteamGames[['item_id', 'name']], on='item_id', how='left')
   
    # Se ordena el DataFrame 'merged_reviews' según los valores de la columna 'sentiment_analysis' de forma descendente.
    df_ordenado = merged_reviews.sort_values(by='sentiment_analysis', ascending=False)

    
    # Se selecciona el top 3 de juegos más recomendados, estos estan ordenados de mayor a menor por el value_counts
    top_3_worst_games = df_ordenado.head(3)

    # Se retorna el resultado
    result = [
        {"Puesto 1": top_3_worst_games.iloc[0]['name']},
        {"Puesto 2": top_3_worst_games.iloc[1]['name']},
        {"Puesto 3": top_3_worst_games.iloc[2]['name']}
    ]
    return result


@app.get('/top3_menos_recomendadas/{anio}')
def Usersworstdeveloper(año:int):
    # Se filtra el DataFrame resultante para el año específico
    reduce_df = df_UserReviews[(df_UserReviews['year'] == año) &
                                 (df_UserReviews['recommend'] == False)  &
                                 (df_UserReviews['sentiment_analysis'] == 0)]
    
    # Se hace un join entre df_UserReviews y df_steamgames utilizando 'item_id' como clave
    merged_reviews = pd.merge(reduce_df, df_SteamGames[['item_id','developer']], on='item_id', how='left')
   
    # Se ordena el DataFrame 'merged_reviews' según los valores de la columna 'sentiment_analysis' de forma descendente.
    df_ordenado = merged_reviews.sort_values(by='sentiment_analysis', ascending=True)

    
    
    top_3_worst_games = df_ordenado.head(3)

    # Se retorna el resultado    
    result = [
        {"Puesto 1": top_3_worst_games.iloc[0]['developer']},
        {"Puesto 2": top_3_worst_games.iloc[1]['developer']},
        {"Puesto 3": top_3_worst_games.iloc[2]['developer']}
    ]
    return result





def convert_numpy_int64(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.ndarray,)):
        return [convert_numpy_int64(item) for item in obj]
    elif isinstance(obj, (dict,)):
        return {key: convert_numpy_int64(value) for key, value in obj.items()}
    elif isinstance(obj, (list,)):
        return [convert_numpy_int64(item) for item in obj]
    else:
        return obj
@app.get('/cantidad_reseñas/{desarrolladora}')
def sentiment_analysis(desarrolladora:str):
        # Se hace un join entre df_UserReviews y df_games utilizando 'item_id' como clave
        merged_reviews = pd.merge(df_UserReviews,df_SteamGames[['item_id','developer']], on='item_id', how='left')

        # Se filtra el DataFrame de reseñas por la desarrolladora específica
        reviews_by_developer = merged_reviews[merged_reviews['developer'] == desarrolladora]

        # Se realiza el análisis de sentimiento
        sentiment_counts = reviews_by_developer['sentiment_analysis'].value_counts()

        # Se crea el diccionario de resultado
        result = {desarrolladora: {
            'Negative': sentiment_counts.get(0, 0),
            'Neutral': sentiment_counts.get(1, 0),
            'Positive': sentiment_counts.get(2, 0)
        }}
        # Convierte los valores numpy.int64 a tipos de datos nativos de Python
        result = convert_numpy_int64(result)

        return JSONResponse(content=result)


# Se filtra el DataFrame original para obtener solo las columnas relevantes
game_columns = ['item_id', 'name']

# Se crea un nuevo DataFrame llamado df_games que contiene solo las columnas especificadas en game_columns.
df_games = df_SteamGames[game_columns]

# Se hace un join entre df_games y df_UserItems utilizando 'item_id' como clave
df_joined = pd.merge(df_games, df_UserReviews[['item_id', 'review']], on='item_id', how='inner')

#Se crea una muestra para el modelo
muestra = df_joined.head(10000)

# Se crea el modelo de machine learning con Scikit-Learn
tfidf = TfidfVectorizer(stop_words='english')
muestra=muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(muestra['review'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)

@app.get('/recomendacion_juego/{id}')
def recomendacion_juego(id: int):
    if id not in muestra['item_id'].values:
        return {'mensaje': 'No existe el id del juego.'}
    
    titulo = muestra.loc[muestra['item_id'] == id, 'name'].iloc[0]
    idx = muestra[muestra['name'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = muestra['name'].iloc[sim_ind].values.tolist()
    return {'juegos recomendados': list(sim_juegos)}