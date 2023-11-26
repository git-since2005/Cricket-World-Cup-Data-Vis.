import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


teams = ['Rajasthan Royals', 'Royal Challengers Bangalore',
         'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings',
         'Gujarat Titans', 'Lucknow Super Giants', 'Kolkata Knight Riders',
         'Punjab Kings', 'Mumbai Indians']

cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
          'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
          'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
          'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
          'Dharamsala', 'Kochi', 'Nagpur', 'Johannesburg', 'Centurion',
          'Durban', 'Bloemfontein', 'Port Elizabeth', 'Kimberley',
          'East London', 'Cape Town']

LogReg = pickle.load(open('LogReg.pkl', 'rb'))
Rf = pickle.load(open('Rf.pkl', 'rb'))
dt_clf = pickle.load(open('dt_clf.pkl', 'rb'))
st.title('The Winning IPL Team Forecast App')


# Load data
@st.cache(allow_output_mutation=True)
def load_data():
    matches_till_2022 = pd.read_csv('Dataset/Processed_Data/Matches_Till_2022.csv')
    return matches_till_2022


match = load_data()

match['WinningTeam'] = match['WinningTeam'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['WinningTeam'] = match['WinningTeam'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match['WinningTeam'] = match['WinningTeam'].str.replace('Kings XI Punjab', 'Punjab Kings')
match['WinningTeam'] = match['WinningTeam'].str.replace('Gujarat Lions', 'Gujarat Titans')
match['WinningTeam'] = match['WinningTeam'].str.replace('Pune Warriors', 'Rising Pune Supergiant')
match['WinningTeam'] = match['WinningTeam'].str.replace('Rising Pune Supergiants', 'Rising Pune Supergiant')


st.image('./Images/Tata_IPL.jpg')

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Random Forest', 'Decision Tree')
)

season = st.sidebar.selectbox(
    'Select Season',
    (2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
     2022))

sel_team = st.sidebar.selectbox(
    'Select Team Name',
    teams)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting Team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling Team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs Completed')

with col5:
    wickets_out = st.number_input('Wicket out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets_out
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({'Batting_Team': [batting_team], 'Bowling_team': [bowling_team],
                             'City': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets_left], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    st.table(input_df)
    if classifier_name == "Logistic Regression":
        result = LogReg.predict_proba(input_df)
    elif classifier_name == "Random Forest":
        result = Rf.predict_proba(input_df)
    else:
        result = dt_clf.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    # st.header(batting_team + "-" + str(round(win*100)) +'%')
    # st.header(bowling_team + "-" + str(round(loss*100)) +'%')

    labels = [batting_team, bowling_team]
    sizes = [win, loss]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(figsize=(2, 2))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Winning Probability of Each Team')
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')

    st.pyplot(fig1)

st.header('Some Stats')

if st.button('Performance of Teams in selected Season'):
    match_wins = match[match['Season'] == season]['WinningTeam'].value_counts()

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x=match_wins.index, y=match_wins.values, alpha=0.8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.title('Performance of Each Team')
    plt.ylabel('Number of Match Wins in the Season', fontsize=12)
    plt.xlabel('Teams', fontsize=12)
    plt.tight_layout()
    # plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

if st.button('Show Team Stats'):
    df = match[(match['WinningTeam'] == sel_team)].groupby(['Season'])['ID'].count()
    dff = match[(match['Team1'] == sel_team) | (
            match['Team2'] == sel_team)].groupby(['Season'])['ID'].count()
    # df = dict(df)
    # dff = dict(dff)
    # st.write(df)
    if season not in df or season not in dff:
        st.error('The team did not play in this season!')
    else:
        win = df[season]
        total = dff[season]

        if win is None:
            st.write("dffff")
        loss = total - win
        data = {'win': [win], 'loss': [loss]}
        data = pd.DataFrame(data)
        st.write(data)
        data.hist()
        st.bar_chart(data=data)
