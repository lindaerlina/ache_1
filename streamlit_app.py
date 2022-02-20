import streamlit as st
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image

st.markdown('# 💊 AChEpred')
st.info('Prediction of Acetylcholinesterase inhibitors and non-inhibitors')

# 1. Load dataset
st.markdown('## 1. Load dataset')
st.info('''
A dataset consisting of Acetylcholinesterase bioactivity data was compiled from the ChEMBL database.

Each compounds were labeled as inhibitors (pIC50 ≥ 6) or non-inhibitors (pIC50 ≤ 5) on the basis of their bioactivity data values.
''')

dataset_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/acetylcholinesterase_07_bioactivity_data_2class_pIC50_pubchem_fp.csv'
dataset = pd.read_csv(dataset_url)

with st.expander('See: Dataset'):
  st.write(dataset)

# 2. Data pre-processing
st.markdown('## 2. Data pre-processing')
          
# Prepare class label column
st.markdown('#### Prepare class label column')
bioactivity_threshold = []
for i in dataset.pIC50:
  if float(i) <= 5:
    bioactivity_threshold.append("inactive")
  elif float(i) >= 6:
    bioactivity_threshold.append("active")
  else:
    bioactivity_threshold.append("intermediate")
    
# Add class label column to the dataset DataFrame
bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df = pd.concat([dataset, bioactivity_class], axis=1)

with st.expander('See: Dataset (with class label column)'):
  st.write(df)

# Select X and Y variables
st.markdown('#### Select X and Y variables')

X = df.drop(['pIC50', 'class'], axis=1)

def target_encode(val):
  target_mapper = {'inactive':0, 'active':1}
  return target_mapper[val]

Y = df['class'].apply(target_encode)

with st.expander('See: X variables'):
  st.write(X)

with st.expander('See: Y variable'):
  st.write(Y)

# Remove low variance features
st.markdown('#### Remove low variance features')

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)

with st.expander('See: X variables (low variance features removed)'):
  st.write(X)

# 3. Random Forest Classification Model
st.markdown('## 3. Random Forest Classification Model')

# Data splitting
st.markdown('#### Data splitting')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

with st.expander('See: X_train, y_train dimensions'):
  st.write(X_train.shape, y_train.shape)
with st.expander('See: X_train, y_train dimensions'):
  st.write(X_test.shape, y_test.shape)
  
# Model Building
st.markdown('#### Model Building')

model = RandomForestClassifier(n_estimators=500, random_state=42)
with st.spinner('Model is building...'):
  model.fit(X_train, y_train)
st.success('Model is trained!')

# Apply Model to Make Predictions
st.markdown('#### Apply Model to Make Predictions')

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Prepare DataFrame of predictions for y_train
df_y_train = pd.concat([pd.Series(list(y_train), name = 'y_train'), pd.Series(y_train_pred, name = 'y_train_pred')], axis=1)

with st.expander('See: Actual vs Predicted Y values for Training set'):
  st.write(df_y_train)
  
# Prepare DataFrame of predictions for y_test
df_y_test = pd.concat([pd.Series(list(y_test), name = 'y_test'), pd.Series(y_test_pred, name = 'y_test_pred')], axis=1)

with st.expander('See: Actual vs Predicted Y values for Test set'):
  st.write(df_y_test)

# 4. Model Performance
st.markdown('## 4. Model Performance')

# Compute the model performance
ac_train = accuracy_score(y_train, y_train_pred)
ac_test = accuracy_score(y_test, y_test_pred)

sn_train = recall_score(y_train, y_train_pred)
sn_test = recall_score(y_test, y_test_pred)

sp_train = recall_score(y_train, y_train_pred, pos_label=0)
sp_test = recall_score(y_test, y_test_pred, pos_label=0)

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_test = matthews_corrcoef(y_test, y_test_pred)

# Prepare a DataFrame of the model performance
metrics = ['Ac', 'Ac', 'Sn', 'Sn', 'Sp', 'Sp', 'MCC', 'MCC']
dataset = ['Train', 'Test', 'Train', 'Test', 'Train', 'Test', 'Train', 'Test']
performance = [ac_train, ac_test, sn_train, sn_test, sp_train, sp_test, mcc_train, mcc_test]

dictionary_performance = {'Metrics': metrics, 'Dataset': dataset, 'Performance': performance}
df_performance = pd.DataFrame(dictionary_performance)

with st.expander('See: Summary Table of Model Performance'):
  st.write(df_performance)

# 5. Confusion matrix
st.markdown('## 5. Confusion matrix')

titles_options = [
    ("Testing Set (Not Normalized)", None),
    ("Testing Set (Normalized)", "true")]

for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=y_test.unique(),
        cmap=plt.cm.Greens,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    with st.spinner('Creating plot...'):
      with st.expander('See: Confusion Matrix'):
        plt.savefig('plot.png', dpi=300)
        image = Image.open('plot.png')
        st.image(image)
