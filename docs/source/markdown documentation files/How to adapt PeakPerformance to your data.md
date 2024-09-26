# How to adapt PeakPerformance to your methods / data
If your data shows a lack of fit when using the default models and prior probability distributions (priors) implemented in PeakPerformance, there are two main scenarios which differ with regards to the action required:

1. An implemented model should be able to fit your peak data, but it does not converge properly.
2. No currently implemented model is able to fit your peak data.

## Scenario 1: Update the model priors
Open the `pipeline.py` file (containing the `pipeline` module) and search for functions starting with "define_".
These will be the functions defining and returning a model object.
Search for the function creating the model that should be able to fit your data and check the priors for its parameters.
An easy way to visualize them is using the `stats` module from the `scipy` package in a seperate notebook and plotting the priors for an example peak of yours.
Most likely, you will find that one or more priors do not represent your data well, e.g. the noise prior since that is highly dependent on the experimental method utilized to obtain your data in the first place.
If you find priors that improve the model fit, just override the priors in the `pipeline` module.
Be aware, though, that they will be overwritten by updates, so you should save them elsewhere and re-apply them when necessary.

## Scenario 2: Implement another model
To implement a new model to PeakPerformance, you will need to create the following functions:

- In the `models` module: a function to define your model calling upon a separate function defining the posterior probablity distribution.
- Optional: create unit tests for your model in `test_models.py`.
- Required by AGPL: Publish your changes open source, for example by uploading the modified code to a public GitHub project. We are also welcoming pull requests that make PeakPerformance more enjoyable to use for everybody.


Additionally, you will need to update the following code segments:

- In the `models` module: update the ModelType class with your model.
- In the `pipeline` module:
  - Add your model to if-statement around the model definition in the `pipeline_loop()` function.
  - Add your model to the `selection_loop()` function.
- Required by AGPL: Publish your changes open source, for example by uploading the modified code to a public GitHub project. We are also welcoming pull requests that make PeakPerformance more enjoyable to use for everybody.
