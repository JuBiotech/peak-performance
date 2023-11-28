# How to adapt PeakPerformance to your methods / data
If your data shows a lack of fit when using the default models and prior probability distributions (priors) implemented in PeakPerformance, there are two main scenarios which differ with regards to the action required:
  1. An implemented model should be able to fit your peak data, but it does not converge properly.
  2. No currently implemented model is able to fit your peak data.

## Scenario 1: Update the model priors
Open the `pipeline.py` file (containing the `pipeline` module) and search for functions starting with "define_". These will be the functions defining and returning a model object. Search for the function creating the model that should be able to fit your data and check the priors for its parameters. An easy way to visualize them is using the `stats` module from the `scipy` package in a seperate notebook and plotting the priors for an example peak of yours. Most likely, you will find that one or more priors do not represent your data well, e.g. the noise prior since that is highly dependent on the experimental method utilized to obtain your data in the first place. If you find priors that improve the model fit, just override the priors in the `pipeline` module. Be aware, though, that they will be overwritten by updates, so you should save them elsewhere and re-apply them when necessary.

## Scenario 2: Implement another model
To implement a new model to PeakPerformance, you will need to create the following functions:  
  - in the `models` module: a function to define your model calling upon a separate function defining the posterior probablity distribution
  - optional: create unit tests for your model in `test_models.py`


Additionally, you will need to update the following code segments:  
  - in the `models` module: update the ModelType class with your model
  - in the `pipeline` module:
    - add your model to if-statement around the model definition in the `pipeline_loop()` function
    - add your model to the `selection_loop()` function
