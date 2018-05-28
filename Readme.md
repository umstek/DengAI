# DengAI

Open `.ipynb` files with jupyter notebook or alternative.  
Run `Preprocess.ipynb` to preprocess source files and generate learn-ready files.  
In jupyterlab, `Run -> Run All` will do this.  
You can tweak it and make changes and try learning with the resulting files in the `generated` folder.  
`ModelSelection.ipynb` is supposed to select the best model to use via a Grid Search Cross Validation (hyperparameter optimization) per each model. But it looks like sklearn is suboptimal (or we don't know how to use it).  
`DengAI.ipynb` is supposed to contain the feature selection, learning and result generation but it has not yet been completed.  

You can use other tools to make predictions.  
Results:  
Matlab Ensemble Boosted Trees with 5-Fold Cross Validation: Error=24.9663  
Settings: Iq -> 7 100 0.09, Sj -> 7 100 0.1  

Please do not push any **changes** (on master) to these files unless the changes reduce the error.  
When you are pushing a notebook, please clear all outputs. e.g.: `Edit -> Clear All Outputs`.  

