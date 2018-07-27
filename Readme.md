# DengAI

## Reports and Presentations  
### [Presentation](https://github.com/umstek/DengAI/blob/master/DengAI.pdf) for CS4622 (Machine Learning)  

### [Report](https://github.com/umstek/DengAI/blob/master/Machine%20Learning%20Report%20-%20Group%2030.pdf) for CS4622 (Machine Learning)  

### [Report](https://github.com/umstek/DengAI/blob/master/Data%20Mining%20Report%20-%20Group%2030.pdf) for CS4642 (Data Mining and Information Retrieval)  


## Results  
Current best result: 19.3798 (MAE), Rank 89 as of July 27 - 2018.  


## Directory contents  
+ The `.` root directory contains the data files downloaded from _drivendata_ and some milestone submissions.  
+ `deprecated` folder contains the first approaches to the problem with _Matlab regression learner_ and _Orange3_ (with minimal preprocessing) and the resulting `.csv` files.  
+ `Neural Networks` folder contains the first approaches to the problem with deep neural networks with _Keras_ and _Tensorflow_.  
+ `Negative Binominal Regression` contains the DengAI benchmark model built with _Jupyter Notebook_ and _sklearn_, _statsmodels_ etc.  
+ `Interactive Python 1` contains the approaches that do general preprocessing with _Jupyter Notebook_, _pandas_, _sklearn_, _statsmodels_, _seaborn_ and uses various models for prediction.  
+ `Interactive Python 2` contains a pipeline that processes the files in various stages using _Jupyter Notebook_, _pandas_, _sklearn_, _statsmodels_, _seaborn_, and _R_'s STL (time series decomposition) borrowed with the _r2py_ bridge. This pipeline does preprocessing, visualization, analysing, automatic selection of features, best model selection etc. The best working model is a time series decomposing predicter with a linear regression model.  
+ `Orange` folder contains an Orange3 pipeline that tests cross-validated errors of various learners with preprocessing, feature engineering etc.  

