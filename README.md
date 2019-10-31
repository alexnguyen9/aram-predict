# aram-predict
Using machine learning to predict League of Legends ARAM matches and scoring the best to worst champions in ARAM.

I deployed this project using pythonanywhere.com, you can check it out [here](http://arampredict.pythonanywhere.com/)

For more information on what I did, you can check out the accompanying [report](https://alexnguyen9.github.io/project/lolaram/)

Steps I performed in this project:
1. Extract sample matches from the riot API
2. Get the champions present in each match and create a (sparse) dataframe
3. Train a logistic regression classifier onto the data
4. Deploy onto flask!

### Step 1: Extracting sample matches
The notebook `Data Extraction.ipynb` contains the code that I used to extract sample ARAM matches. To get a list of champions and their associated key (the match data contains the key values not the actually champion names), we use the json file `champion.json` taken from the riot website.  To convert a list of champions into a p-dimensional vector (where p is the number of champions) we use sklearn's CountVectorizer,  this will be used to convert the teams into vectors.  

This produces: `countvectorizer.p`

To extract data from riot, you will need to have a league of legends account to get an KPI key, check out [here](https://developer.riotgames.com/) to get a key and enter it into the notebook.  The notebook exports the sample match ID's into a sorted list that is saved as a pickle file:  `pulled_match_ids.p`

I primarily used the [Cassiopeia package](https://github.com/meraki-analytics/cassiopeia) to extract the sample matches.  

### Step 2: Get Champions & Create Dataframe
The second portion in the `Data Extraction.ipynb` notebook takes each the match ID's and finds the champions in each team present in that particular match, and which team won (which would be our target) and creates a dataframe (columns represent champions, rows represent a particular match) using the count vectorizer from above. 

This exports the final dataframe as a  pickle file `final_df.p`.

### Step 3: 
The python file  `train_and_scoring.py` takes in the final dataframe and trains the data using logistic regression, ridge classifier, SVM, and decision trees. A simple cross validation was utilized to tune each of the 4 models. For my project, I utilized the logistic regression classifier.  

This produces the pickle file `logistic_regression.p` as the fitted model.

Secondly, I produced a dataframe that contained a sorted dataframe of the champions using their coefficients (I converted it into a 0 to 100 range) 

This produces the pickled dataframe: `scores.p`,
which will be used in the flask app

#### Step 4:
To deploy locally make sure you have 
* `champions.json`
* `countvectorizer.p`
* `logistic_regression.p`
* `scores.p` 

Then simply run `app.py`
