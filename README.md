# Predicting next purchase month

<p align="center">
  <img src="data/figures/accuracy.png" width="400">
  <img src="data/figures/precision.png" width="400">
</p>
<p align="center"><img src="data/figures/single_month.png" width=800></p>

Using the same dataset and daily aggregation statistics as in the [Customer Segmentation project](https://github.com/W-Tran/online-retail), I instead built a ML classifier to predict whether a particular customer will purchase within the next month. The data is split into holdout and calibration sets, akin to time series cross validation, to avoid data leakage. 
 
Specifically, I selected cut-off points at the beginning of each month which I call observation end dates. All transactions before the observation end date are used to train the model and build the test features, whilst the transactions after the observation end date are used as the test labels, 1 if the customer purchased within the next month, 0 otherwise.

The model is compared to a naive baseline which predicts that all customers with a purchase frequency of less than a month will purchase again within the next month. A small lift in accuracy over the naive model is observed throughout the first and second year, however there is considerable lift over the naive model in precision after 1 year of data is used to train the model. 

The rationale behind evaluating using precision is that we care most about True Positives, a customer purchasing when we think they will, as it represents anticipation of voluntary customer engagement. We can perhaps use this information in order to act to prevent churn for high risk customers who are still "alive" or attempt to promote a low/medium value customer to a high value customer using marketing tactics or by other means.


**Notes/Future work**

- Item transaction history i.e. what exactly a customer is buying could have some predictive power. Explore using categorical embeddings to encode items into features that may boost the performance of the ML model.
- Probabilistic [BTYD](https://en.wikipedia.org/wiki/Buy_Till_you_Die) [models](https://lifetimes.readthedocs.io/en/latest/) were explored extensively for use in this project. However most available packages were not able to deal with the heavy seasonality in the purchase behaviour of customers for this particular online retail (a gift shop which sees a huge surge in sales leading up to Christmas). Fitting BTYD model parameters without incorporating seasonality results in poor performance.
- [Discrete Survival models](https://data.princeton.edu/wws509/notes/c7s6) were also explored for this project (inspired by [this talk](https://www.youtube.com/watch?v=uU1u6JQCg5U)) but also failed to perform well in predicting purchase behaviour. I also intend to study survival models a bit more in the future to explore ways in which they can be used for online retail settings.


## Installation

Simply clone the repo and install all dependencies listed in the requirements.txt file to an environment of your choice.

## Usage

All results and plots can be reproduced using the predicting_purchases_in_next_month.ipynb notebook.
