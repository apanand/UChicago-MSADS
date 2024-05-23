# Spotify Song Recommender using a Simple Neural Network


This project leverages the Spotify API to create a simple neural network recommendation system for myself. Using my listening history and metadata acquired using the Spotipy packagae in Python, I have trained a simple Neural Network with several hidden layers to recommend songs on the Billboard Top 100 as of May 2024 to myself. 

The [data](https://github.com/apanand/UChicago-MSADS/tree/main/Spotify%20Recommender/data) folder holds my listening history as provided by requesting Spotify in advance. It comes in 4 JSON files containing all of the songs I listened to as well as artist of the song and the duration I played the song for. 

I have also included two dataframes in there, one for my listening history and one for the billboard top 100 as of May 2024. More on this below. 

There are two Google Colab ipynb notebooks here: one for data acquisition and wrangling and the other for the actual modeling.

Roadmap:

1. In order to get the data in usable form, I had to scrape the Spotify API for metadata. This code can be found in the [Data_Acquisition_and_Wrangling.ipynb](https://github.com/apanand/UChicago-MSADS/blob/main/Spotify%20Recommender/Data_Acquisition_and_Wrangling.ipynb) file. This code gathers the data that goes into the listening_history.csv and billboard_features_df.csv files.

2. After scraping the data with the Spotify API and creating those two dataframes, the [Modeling.ipynb](https://github.com/apanand/UChicago-MSADS/blob/main/Spotify%20Recommender/Modeling.ipynb) file contains Exploratory Data Analysis as well as the Neural Network model I have written to craft recommendations to myself. The simple neural network is near the bottom as well as results calculations (test and train accuracy, loss, etc.)


If you have any questions or encounter any bugs, feel free to email me at: [apanand@uchicago.edu](apanand@uchicago.edu)





![University of Chicago MS in Applied Data Science Logo](https://voices.uchicago.edu/msca2/files/2023/06/UChicago_PSD_MS-AppliedDataScience_Vertical_Color-RGB-1.png)


![Spotify Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Spotify_logo_with_text.svg/1200px-Spotify_logo_with_text.svg.png)
