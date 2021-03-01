import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import hex_to_rgb
from plotly.subplots import make_subplots
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#------------------------------------------------------------------------------------------Definitions-Notebook
def create_soup(data):
    att = data['Title'].lower()
    for i in data[1:]:
            att = att + ' ' + str(i)
            return att

# Function to plot value counts plots
def plot_value_counts_bar(data, col):

    vc = pd.DataFrame(data[col].value_counts())
    vc['cat'] = vc.index
    fig = px.bar(vc, x='cat', y=col, color='cat', title=col)
    fig.update_layout()

    return fig

# Repolacing % with a none value
def str_to_int(val):
    new_val = re.sub('%', '', val)
    return (int(new_val))


# Rounding the ratings for EDA
def round_fix(data):
    data_str = str(data).strip()
    if data_str != 'NA':
        data = str_to_int(data_str)
        if data in range(0, 40):
            return 'Really_Bad'
        if data in range(41, 55):
            return 'Bad'
        if data in range(56, 70):
            return 'Average'
        if data in range(71, 85):
            return 'Good'
        if data in range(85, 101):
            return 'Really_Good'
    else:
        return 'NA'
        
# Rounding the ratings for IMDB
def round_fix_imdb(data):
    if data != 'NA':
        data = float(data)
        out=None
        if data>=0.0 and data<=3.5:
            out= 'Really_Bad'
        if data>=3.6 and data<=5.0:
            out= 'Bad'
        if data>=5.1 and data<=6.5:
            out= 'Average'
        if data>=6.5 and data<=7.9:
            out= 'Good'
        if data>=8.0 and data<=10: 
            out= 'Really_Good'
        return out
    else:
        return 'NA'

# This function will return a list of categories present in the dataset
def get_categories(data):
    test = data[1]
    for i in data[1:]:
        if str(i).strip() != 'nan':
            test = test + ',' + i
    test = test.split(',')
    return list(set(test))


# To keep all the categories for every column update
def keep_categories(data, columns):
    kata = {column: None for column in columns}
    for column in columns:
        kata[column] = get_categories(data[column])
    return kata


# This function will return the count of each category from the dataset
def get_cat_order(data, col):
    cat = get_categories(data)
    # Create a dictionary to store the values
    cata = {
        col + '_' + category: [0 for i in range(len(data))]
        for category in cat
    }
    element_index = 0
    for element in data:
        element = element.split(',')
        for category in element:
            cata[col + '_' + category][element_index] = 1
        element_index += 1
    return cata


# This function will replace the original column with new columns of categories
def update_data(data, col, cata):
    for i in cata:
        data[i] = cata[i]
    data.drop(col, inplace=True, axis=1)
    return data


# This will apply encoding over all the columnns
# If get_kata is 1 then it will call keep_categories in itself and return two values
# Otherwise, only single dataframe will be returned
def apply_encoding(data, columns, get_kata):
    if get_kata == 1:
        kata = keep_categories(data, columns)
    for column in columns:
        cata = get_cat_order(data[column], column)
        data = update_data(data, column, cata)
    if get_kata == 1:
        return kata, data
    else:
        return data

def get_counts(data, col, categories):
    cata = {category: None for category in categories}
    for category in categories:
        cata[category] = data[col + '_' + category].sum()
    return cata

def get_ott_counts(data, platforms, column):
    out = {'ID': [], column: [], 'Count': [], 'Platform': []}
    iid = 0
    for platform in platforms:
        print('>>>> Done: ' + platform)
        years = data[data[platform] == 1].groupby(column)
        for year in years.groups:
            year_grp = years.get_group(year)
            out['ID'].append(iid)
            out[column].append(year)
            out['Count'].append(year_grp.shape[0])
            out['Platform'].append(platform)
            iid += 1
    return pd.DataFrame(out)

def replaceNAby1(rating):
    if rating=='NA':
        return '-1'
    else:
        return rating

def color_platform(platform):
    if platform == 'Netflix':
        return ['#6F0000', '#FF0000']
    elif platform == 'Prime Video':
        return ['#06064D', '#1616CD']
    elif platform == 'Hulu':
        return ['#00DE00', '#005800']
    else:
        return ['#00BDBD', '#004242']


def plot_genres(genres, platform,temp_data):
    tg_data = temp_data.copy()
    if genres == 'All' and platform == 'All':
        title = 'Top 15 Movies/Series'
        tg_data.sort_values(by='IMDb', ascending=False, inplace=True)
        fig = px.bar(tg_data[:15],
                     y='Title',
                     x='IMDb',
                     color='IMDb',
                     title=title,
                     color_continuous_scale=['#E6009B', '#5E003F'],orientation='h')
    elif genres == 'All' and platform != 'All':
        cequence = color_platform(platform)
        title = 'Top 15 Movies/Series on ' + platform
        tg_data = tg_data[tg_data[platform] == 1]
        tg_data.sort_values(by='IMDb', ascending=False, inplace=True)
        fig = px.bar(tg_data[:15],
                     y='Title',
                     x='IMDb',
                     color='IMDb',
                     title=title,
                     color_continuous_scale=cequence,orientation='h')
    elif genres != 'All' and platform == 'All':
        title = 'Top 15 ' + genres + ' Movies/Series'
        tg_data = tg_data[(tg_data['Genres_' + genres] == 1)]
        tg_data.sort_values(by='IMDb', ascending=False, inplace=True)
        fig = px.bar(tg_data[:15],
                     y='Title',
                     x='IMDb',
                     color='IMDb',
                     title=title,
                     color_continuous_scale=['#F52668', '#6D0023'],orientation='h')
    else:
        cequence = color_platform(platform)
        title = 'Top 15 ' + genres + ' Movies/Series on ' + platform
        tg_data = tg_data[(tg_data[platform] == 1)
                          & (tg_data['Genres_' + genres] == 1)]
        tg_data.sort_values(by='IMDb', ascending=False, inplace=True)
        fig = px.bar(tg_data[:15],
                     y='Title',
                     x='IMDb',
                     color='IMDb',
                     title=title,
                     color_continuous_scale=cequence,orientation='h')
    return fig

# Let's start processing the age column first
def convertAge(age):
    if age!='NA':
        if age=='18+':
            return 'Adult'
        elif age=='13+' or age=='16+':
            return 'Teen'
        else:
            return 'Non-Adult'
    else:
        return age

# This function will return the count of each category from the dataset
def encode_data(data,cat,col):
    # Create a dictionary to store the values
    cata = {
        col + '_' + category: [0 for i in range(len(data))]
        for category in cat
    }
    element_index = 0
    for element in data:
        element = element.split(',')
        for category in element:
            try:
                cata[col + '_' + category][element_index] = 1
            except KeyError:
                cata[col+'_'+'Others'][element_index]=1
        element_index += 1
    return cata

def get_recommendations_new(title, data, o_data, cosine_sim):
    cbr_data=data.copy()
    data = data.reset_index()
    indices = pd.Series(data.index, index=cbr_data['Title'])
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    return o_data[['Title', 'IMDb', 'Genres', 'Directors', 'Country', 'Language']].iloc[movie_indices]
#------------------------------------------------------------------------------------------Data Preparation
data = pd.read_csv('MoviesOnStreamingPlatforms_updated.csv')
cols = list(data.columns)
cols[0] = 'Temp'
data.columns = cols
data.drop('Temp', axis=1, inplace=True)
data.set_index('ID', inplace=True)
# Filling NA values with NA
data['Rotten Tomatoes'].fillna('NA', inplace=True)
data['Age'].fillna('NA', inplace=True)
data['Directors'].fillna('NA', inplace=True)
data['Language'].fillna('NA', inplace=True)
data['Directors'].fillna(-1, inplace=True)
data['IMDb'].fillna('NA', inplace=True)
data['Country'].fillna('NA', inplace=True)
data['Genres'].fillna('NA', inplace=True)
data.Runtime.fillna(-1, inplace=True)
eata =  pd.read_csv('movies_data_encoded.csv',index_col=0)
eata.fillna('NA',inplace=True)
remove=['Age','Year','Type','Rotten Tomatoes','IMDb','Rotten_Tomatoes_Rounded']
dummy=['Era','AgeRestriction','IMDB_Rounded']
#----------------------------------------------------------------------------------------------Definitions-Streamlit


def perform_eda(data):
    st.title('Data Analysis')

    st.markdown('## Age Analysis')
    st.plotly_chart(plot_value_counts_bar(data, 'Age'))
    st.markdown('''#### Observation: 
    - Most of the movies/shows are targeted to adult audience''')

    st.markdown('## Rotten Tomatoes Ratings')
    data['Rotten_Tomatoes_Rounded'] = data['Rotten Tomatoes'].apply(round_fix)
    st.plotly_chart(plot_value_counts_bar(data, 'Rotten_Tomatoes_Rounded'))
    st.markdown('''#### Observations:
- Most of the content is high rated on streaming platforms''')

    st.markdown('## IMDB Ratings')
    data['IMDB_Rounded'] = data.IMDb.apply(round_fix_imdb)
    st.plotly_chart(plot_value_counts_bar(data, 'IMDB_Rounded'))
    st.markdown('''#### Observations:
- Most of content on streaming platforms has average ratings on IMDB.''')


    st.markdown('## Highest IMDb Movies/Shows')
    netflix_count = data[data['IMDB_Rounded']=='Really_Good']['Netflix'].sum()
    hulu_count = data[data['IMDB_Rounded']=='Really_Good']['Hulu'].sum()
    disney_count = data[data['IMDB_Rounded']=='Really_Good']['Disney+'].sum()
    prime_count = data[data['IMDB_Rounded']=='Really_Good']['Prime Video'].sum()
    indexes = ['Netflix', 'Hulu', 'Disney', 'Amazon Prime']
    values = [netflix_count, hulu_count, disney_count, prime_count]
    fig=px.pie(labels=indexes, values=values,title='Top content on OTT',hover_name=indexes)
    st.plotly_chart(fig)


    st.markdown('## Most Popular Genre')
    temp_data=data.copy()
    kata, temp_data = apply_encoding(temp_data, ['Genres', 'Country', 'Language'], get_kata=1)
    base_counts = get_counts(temp_data, 'Genres', kata['Genres'])
    base_counts = pd.DataFrame(index=base_counts.keys(),
                            data=base_counts.values(),
                            columns=['Counts'])
    base_counts.sort_values(by='Counts', inplace=True)
    colors=['#988D90' if i<1000 else '#F00045' for i in  base_counts.Counts]
    fig = px.bar(x=base_counts.index,
                y=base_counts['Counts'],
                title='Most Popular Genre',color_discrete_sequence=colors,color=base_counts.index)
    st.plotly_chart(fig)
    st.markdown('''#### Observations:
- Drama is most popular genre''')

    st.markdown('## Most Released Content')
    st.markdown('### Country')
    base_counts = get_counts(temp_data, 'Country', kata['Country'])
    base_counts = pd.DataFrame(index=base_counts.keys(),
                            data=base_counts.values(),
                            columns=['Counts'])
    base_counts.sort_values(by='Counts', ascending=False, inplace=True)
    fig = px.bar(x=base_counts.index[:10],
                y=base_counts['Counts'][:10],
                color=base_counts['Counts'][:10],
                title='Most Released Content')
    st.plotly_chart(fig)
    st.markdown('''#### Observations:
- Most released content was in US''')

    st.markdown('### Language')
    base_counts = get_counts(temp_data, 'Language', kata['Language'])
    base_counts = pd.DataFrame(index=base_counts.keys(),
                            data=base_counts.values(),
                            columns=['Counts'])
    base_counts.sort_values(by='Counts', ascending=False, inplace=True)
    fig = px.bar(x=base_counts.index[:5],
                y=base_counts['Counts'][:5],
                color=base_counts['Counts'][:5],
                title='Most Released Content: Language')
    st.plotly_chart(fig)
    st.markdown('''#### Observations:
From the above visualizations we can conclude that:
- We can work with few genres with count more than 1000 and rest of the genres can be categorized as others.
- It is important to keep countries, but at continent level for better clarity.
- Most of the content is in english only. ''')


    st.markdown('## OTT Platforms')
    st.markdown('### Content Releases')
    release_scores = get_ott_counts(temp_data,
                                ['Netflix', 'Hulu', 'Prime Video', 'Disney+'],
                                'Year')
    fig = px.scatter(
    release_scores,
    x='Year',
    y='Count',
    size='Count',
    color='Platform',
    title='Content Per OTT Apps released in consecutive years',
    color_discrete_sequence=['#E50914', '#3DBB3D', '#00A8E1', '#048f70 '])
    st.plotly_chart(fig)
    st.markdown('''#### Observations:
    - Amazon Prime Video has the most modern as well as old content''')

    st.markdown('### Top Genres')
    genres = kata['Genres'].copy()
    genres.extend(['All'])
    platform = ['Netflix', 'Hulu', 'Prime Video', 'Disney+', 'All']
    temp_data.IMDb=temp_data.IMDb.apply(replaceNAby1)
    temp_data.IMDb=temp_data.IMDb.astype(float)
    genre=st.selectbox('Genres',genres)
    plt_frm=st.selectbox('Platform',platform)
    st.plotly_chart(plot_genres(genre,plt_frm,temp_data))


    keep_genres=pickle.load(open('keep_genres.pickle','rb'))
    genre_counts = get_counts(eata, 'Genres', keep_genres)
    genre_counts = get_counts(eata, 'Genres', keep_genres)
    genre_counts = pd.DataFrame(index=genre_counts.keys(),
                            data=genre_counts.values(),
                            columns=['Counts'])
    genre_counts.sort_values(by='Counts',inplace=True)

    cont_counts = get_counts(eata, 'Continent',
                ['Africa', 'Antarctica', 'Asia', 'Europe', 'North America', 'Oceania', 'South America','NA'])
    cont_counts = pd.DataFrame(index=cont_counts.keys(),
                            data=cont_counts.values(),
                            columns=['Counts'])
    cont_counts.sort_values(by='Counts', ascending=False, inplace=True)
    
    keep_lang=pickle.load(open('keep_lang.pickle','rb'))
    lang_counts = get_counts(eata, 'Language',keep_lang)
    lang_counts = pd.DataFrame(index=lang_counts.keys(),
                            data=lang_counts.values(),
                            columns=['Counts'])
    lang_counts.sort_values(by='Counts', ascending=False, inplace=True)




def perform_recommendation(data):
    st.title('Recommdations')
    cbr_data = eata.copy()
    # Features to be removed or to be encoded
    cbr_data.drop(remove, axis=1, inplace=True)
    cbr_data.drop('Runtime', axis=1, inplace=True)
    cbr_data['soup'] = cbr_data.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(cbr_data['soup'])
    cosine_sim2 = cosine_similarity(count_matrix)
    movie=st.selectbox('Movie/Series',list(cbr_data.Title))
    l = get_recommendations_new(movie, cbr_data, data.copy(), cosine_sim2)
    l.IMDb = l.IMDb.apply(replaceNAby1)
    l.IMDb = l.IMDb.astype(float)
    l.sort_values(by='IMDb',ascending=False,inplace=True)
    st.dataframe(l.sort_values(by='IMDb',ascending=False))

#----------------------------------------------------------------------------------------------Streamlit
out=st.sidebar.radio('Flicky Recommendations',['Data Analysis','Recommendations'])
if out=='Data Analysis':
    perform_eda(data)
else:
    perform_recommendation(data)