All the data preparation steps should be fit using train data. Otherwise, you risk applying the wrong transformations, because means and variances that StandardScaler estimates do probably differ between train and test data.

The easiest way to train, save, load and apply all the steps simultaneously is to use Pipelines:

#### At training:
```python
# prepare the pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

pipe = make_pipeline(StandardScaler(), LogisticRegression)

pipe.fit(X_train, y_train)
joblib.dump(pipe, 'model.pkl')

```
#### At prediction:
```python
#Loading the saved model with joblib
pipe = joblib.load('model.pkl')

# New data to predict
pr = pd.read_csv('set_to_predict.csv')
pred_cols = list(pr.columns.values)[:-1]

# apply the whole pipeline to data
pred = pd.Series(pipe.predict(pr[pred_cols]))
print pred
```