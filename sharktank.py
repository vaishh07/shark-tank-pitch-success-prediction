import streamlit as st
import plotly as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sharktankscience


#loading_data
df = pd.read_csv('Shark Tank India.csv')


#add_header
st.title("Shark Tank Analysis")

#feature_engineering
df['Season Number'] = df['Season Number'].astype(pd.Int32Dtype())
df['Episode Number'] = df['Episode Number'].astype(pd.Int32Dtype())
df['Pitch Number'] = df['Pitch Number'].astype(pd.Int32Dtype())
df['Number of Presenters'] = df['Number of Presenters'].astype(pd.Int32Dtype())
df['Male Presenters'] = df['Male Presenters'].astype(pd.Int32Dtype())
df['Female Presenters'] = df['Female Presenters'].astype(pd.Int32Dtype())
df['Transgender Presenters'] = df['Transgender Presenters'].astype(pd.Int32Dtype())
df['Couple Presenters'] = df['Couple Presenters'].astype(pd.Int32Dtype())
df['Gross Margin'] = df['Gross Margin'].astype(pd.Int32Dtype())
df['Started in'] = df['Started in'].astype(pd.Int32Dtype())
df['Yearly Revenue'] = df['Yearly Revenue'].astype(pd.Int32Dtype())
df['Received Offer'] = df['Received Offer'].astype(pd.Int32Dtype())
df['Accepted Offer'] = df['Accepted Offer'].astype(pd.Int32Dtype())



selected = option_menu(
            menu_title=None,  # required
            options=["Season", "Sharks","Test Your Idea"],  # required
            icons=["house", "man", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
)



if selected == "Season":
    # st.title(f"You have selected {selected}")
    
    add_sidebar=st.sidebar.selectbox('SEASON',('Season 1','Season 2'))
    s1 = df.loc[df['Season Number']== 1]
    s2 = df.loc[df['Season Number']== 2]


    if add_sidebar == 'Season 1' :
        st.title("Season 1 analysis")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        pitch = s1.loc[df['Episode Number']!=0]['Startup Name'].count()
        male=int(s1['Male Presenters'].sum())
        female=int(s1['Female Presenters'].sum())
        rec = (s1['Received Offer']== 1).sum()
        rej = (s1['Received Offer']== 0).sum()
        acc = (s1['Accepted Offer']== 1).sum()
        cacc = (s1['Accepted Offer']== 0).sum()

        col1.metric('Total pitches',pitch,"")
        col2.metric('Offers received',rec," ")
        col3.metric('Accepted offers',acc," ")
        col4.metric('No Offer',rej," ")
        col5.metric('Offer declined',cacc," ")

        
       #fig.update_yaxes(text_title ="")
        
        
        highest = s1.sort_values('Total Deal Amount', ascending=False)[0:40]
        fig = px.bar(highest, x="Startup Name", y='Total Deal Amount', color="Startup Name", title="Highest Investment as per deal amount (in lakhs)", text=highest['Total Deal Amount'].map(int).map(str) + " lakhs")
        st.plotly_chart(fig)

        ind = s1['Industry'].value_counts().sort_values(ascending=True)
        fig = px.bar(ind, x="Industry", title="<b> Industry wise startups")
        fig.update_yaxes(title_text="")
        fig.update_xaxes(visible=False)
        
        st.plotly_chart(fig)
    
    
    


    if add_sidebar == 'Season 2':

        st.title("Season 2 analysis")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        pitch = s2.loc[df['Episode Number']!=0]['Startup Name'].count()
        male=int(s2['Male Presenters'].sum())
        female=int(s2['Female Presenters'].sum())
        rec = (s2['Received Offer']== 1).sum()
        rej = (s2['Received Offer']== 0).sum()
        acc = (s2['Accepted Offer']== 1).sum()
        cacc = (s2['Accepted Offer']== 0).sum()

        col1.metric('Total pitches',pitch,"")
        col2.metric('Offers received',rec," ")
        col3.metric('Accepted offers',acc," ")
        col4.metric('No Offer',rej," ")
        col5.metric('Offer declined',cacc," ")

        
        
        highest = s2.sort_values('Total Deal Amount', ascending=False)[0:40]
        fig = px.bar(highest, x="Startup Name", y='Total Deal Amount', color="Startup Name", title="Highest Investment as per deal amount (in lakhs)", text=highest['Total Deal Amount'].map(int).map(str) + " lakhs")
        st.plotly_chart(fig)

        ind = s2['Industry'].value_counts().sort_values(ascending=True)
        fig = px.bar(ind, x="Industry", title="<b> Industry wise startups")
        fig.update_yaxes(title_text="")
        fig.update_xaxes(visible=False)
        
        st.plotly_chart(fig)







if selected == "Test Your Idea":
    st.subheader("Test your idea")
    df = sharktankscience.clean_data(sharktankscience.df)
    lr = sharktankscience.train_model(df)
    data = pd.DataFrame()    
    # Create a DataFrame from user input
    def predict_offer(lr):
        with st.form(key = 'form1'): 
            # Take input from user
            presenters = st.number_input('Enter Number of Presenters: ')
            revenue = st.number_input('Enter Yearly Revenue: ')
            margin = st.number_input('Enter Net Margin: ')
            ask_amount = st.number_input('Enter Original Ask Amount: ')
            offered_equity = st.number_input('Enter Original Offered Equity: ')
            has_patents = st.number_input('Enter Has Patents (0: No, 1: Yes): ')
            website = st.number_input('Enter Website (0: No, 1: Yes): ')
            years_active = st.number_input('Enter Years Active: ')
            submit_button = st.form_submit_button(label ='View your results')
    
        data = pd.DataFrame({
            'Number of Presenters': [presenters],
            'Yearly Revenue': [revenue],
            'Monthly Sales': [(revenue/12)],
            'Gross Margin' : [margin],
            'Original Ask Amount': [ask_amount],
            'Original Offered Equity': [offered_equity],
            'Valuation Requested': [(ask_amount*offered_equity)],
            'Has Patents': [has_patents],
            'website': [website],
            'years_active': [years_active]
        })

        if submit_button :
            prediction = lr.predict(data)
            if prediction == 1:
                st.success('This startup is likely to receive an offer.')
            else:
                st.success('This startup is unlikely to receive an offer.')
    
    predict_offer(lr)   

    
    
    


    


    



if selected == "Sharks":
    #st.title(f"You have selected {selected}")

    add_sidebar2 = st.sidebar.selectbox('SHARKS',('Vineeta','Aman','Namita','Ashneer','Peyush','Anupam'))

    st.title("Investment by Sharks")

    if add_sidebar2 == "Vineeta" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Vineeta Investment Amount'].sum()/100, 2)
        deals = df[df['Vineeta Investment Amount']>=0][['Vineeta Investment Amount']].count().to_string()[-2:]
        deb = round(df['Vineeta Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        vineeta = df.loc[df['Vineeta Investment Amount']>0]
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Vineeta Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = vineeta.loc[vineeta['Vineeta Investment Amount']>=0] [["Startup Name","Vineeta Investment Amount","Vineeta Investment Equity"]].sort_values(by="Vineeta Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Vineeta Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)
    
        st.dataframe(df.loc[df['Vineeta Investment Amount']>0][["Startup Name","Industry","Vineeta Investment Amount"]])
        

    if add_sidebar2 == "Aman" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Aman Investment Amount'].sum()/100, 2)
        deals = df[df['Aman Investment Amount']>=0][['Aman Investment Amount']].count().to_string()[-2:]
        deb = round(df['Aman Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        aman = df.loc[df['Aman Investment Amount']>0]
        
        
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Aman Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = aman.loc[aman['Aman Investment Amount']>=0] [["Startup Name","Aman Investment Amount","Aman Investment Equity"]].sort_values(by="Aman Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Aman Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)


    if add_sidebar2 == "Anupam" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Anupam Investment Amount'].sum()/100, 2)
        deals = df[df['Anupam Investment Amount']>=0][['Anupam Investment Amount']].count().to_string()[-2:]
        deb = round(df['Anupam Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        anupam = df.loc[df['Anupam Investment Amount']>0]
        
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Anupam Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = anupam.loc[anupam['Anupam Investment Amount']>=0] [["Startup Name","Anupam Investment Amount","Anupam Investment Equity"]].sort_values(by="Anupam Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Anupam Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)
          

    if add_sidebar2 == "Ashneer" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Ashneer Investment Amount'].sum()/100, 2)
        deals = df[df['Ashneer Investment Amount']>=0][['Ashneer Investment Amount']].count().to_string()[-2:]
        deb = round(df['Ashneer Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        ashneer = df.loc[df['Ashneer Investment Amount']>0]
        
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Ashneer Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = ashneer.loc[ashneer['Ashneer Investment Amount']>=0] [["Startup Name","Ashneer Investment Amount","Ashneer Investment Equity"]].sort_values(by="Ashneer Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Ashneer Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)


    if add_sidebar2 == "Peyush" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Peyush Investment Amount'].sum()/100, 2)
        deals = df[df['Peyush Investment Amount']>=0][['Peyush Investment Amount']].count().to_string()[-2:]
        deb = round(df['Peyush Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        peyush = df.loc[df['Peyush Investment Amount']>0]
        
       
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Peyush Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = peyush.loc[peyush['Peyush Investment Amount']>=0] [["Startup Name","Peyush Investment Amount","Peyush Investment Equity"]].sort_values(by="Peyush Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Peyush Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)

    if add_sidebar2 == "Namita" :
        col1, col2, col3 = st.columns(3)
        inv = round(df['Namita Investment Amount'].sum()/100, 2)
        deals = df[df['Namita Investment Amount']>=0][['Namita Investment Amount']].count().to_string()[-2:]
        deb = round(df['Namita Debt Amount'].sum()/100, 2)
        col1.metric("Total Investment", inv , "crores")
        col2.metric("Total Deals", deals , "startups")
        col3.metric("Debt", deb, "crores")
        namita = df.loc[df['Namita Investment Amount']>0]
        
        pie_chart = px.pie(df,
                   title=f"Industry wise inestments",
                   values="Namita Investment Amount",
                   color="Industry",
                   color_discrete_map={ 'amount_wertvoll_1':'darkgreen',
                                        'amount_gut_2':'lime',
                                        'amount_okay_3':'yellow',
                                        'amount_schlecht_4':'orange',
                                        'amount_grottig_5':'red'},
                   names="Industry"
                  )

        st.plotly_chart(pie_chart)
    
        st.header("Company details:")
    
        tmpdf = namita.loc[namita['Namita Investment Amount']>=0] [["Startup Name","Namita Investment Amount","Namita Investment Equity"]].sort_values(by="Namita Investment Equity")
        fig = px.treemap(tmpdf, path=['Startup Name'], values=tmpdf['Namita Investment Amount'], width=800, height=800)
        fig.update_layout(margin = dict(t=5, l=5, r=5, b=5))
        fig.update_traces(textposition='middle center')
        st.plotly_chart(fig)