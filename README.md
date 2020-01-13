# explain_loan_probabiity_of_default_SHAP_values

explain_loan_probabiity_of_default_with_shap_values_python

Some models are easy to interpret such as linear / logistic regression (weight on each feature, knowing the exact contribution and negative and positive interaction), single decision trees, some models are harder to interpret such as Ensemble models - it is hard to understand the role of each feature, it comes with "feature importance" but does not tell if feature affects decision positively or negatively

The Shapley value is the average of all the marginal contributions to all possible coalitions. If we estimate the Shapley values for all feature values, we get the complete distribution of the prediction (minus the average) among the feature values.The interpretation of the Shapley value for feature value j is: The value of the j-th feature contributed ϕj to the prediction of this particular instance compared to the average prediction for the dataset.
