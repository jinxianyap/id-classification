Automated training pipeline built to for performance evaluation and comparison of several multi-class classification CNN models.

# id-classification
`models.py`: compiles and trains selected model, saves model and weights, saves testing and training set prediction results.  
`metrics.py`: pulls saved results from all models specified to output confusion matrix for each model, and comparison of accuracy among all models.

## Usage
`
python models.py <name of model to train>`

`python metrics.py <model1> [<model2> ...]
`
