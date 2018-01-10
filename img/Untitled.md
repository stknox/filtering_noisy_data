# Adult Data Set 

Download: [Data Folder](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/), [Data Set Description](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names)

https://archive.ics.uci.edu/ml/datasets/adult

Abstract: Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

**Given**: This dataset performs very well with linear models. 

<table border="1" cellpadding="6">
    <tr>
		<td bgcolor="#DDEEFF"><b>Data Set Characteristics:</b></td>
		<td>Multivariate</td>
		<td bgcolor="#DDEEFF"><b>Number of Instances:</b></td>
		<td>48842</td>
		<td bgcolor="#DDEEFF"><b>Area:</b></td>
		<td>Social</td>
	</tr>
	<tr>
		<td bgcolor="#DDEEFF"><b>Attribute Characteristics:</b></td>
		<td>Categorical, Integer</td>
		<td bgcolor="#DDEEFF"><b>Number of Attributes:</b></td>
		<td>14</td>
		<td bgcolor="#DDEEFF"><b>Date Donated</b></td>
		<td>1996-05-01</td>
	</tr>
	<tr>
		<td bgcolor="#DDEEFF"><b>Associated Tasks:</b></td>
		<td>Classification</td>
		<td bgcolor="#DDEEFF"><b>Missing Values?</b></td>
		<td>Yes</td>
		<td bgcolor="#DDEEFF"><b>Number of Web Hits:</b></td>
		<td>1008052</td>
	</tr>
</table>

## Attribute Information:

Listing of attributes: 

- `class`: >50K, <=50K. 
- `age`: continuous. 
- `workclass`: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
- `fnlwgt`: continuous. 
- `education`: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
- `education-num`: continuous. 
- `marital-status`: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
- `occupation`: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
- `relationship`: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
- `race`: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
- `sex`: Female, Male. 
- `capital-gain`: continuous. 
- `capital-loss`: continuous. 
- `hours-per-week`: continuous. 
- `native-country`: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Problem Statement

> Definition: A computer program is said to learn from experience `E` with respect to some class of tasks `T` and performance measure `P`, if its performance at tasks in `T`, as measured by `P`, improves with experience `E`.

From Mitchell's *Machine Learning*

Here we will call the known information about the adults in this data set to be `E`.

We will define our task `T` to be a binary classification of whether or not the individuals income exceeds $50K/yr.

We will define a performance metric `P` to be an accuracy score, not an F-score. 

**We seek a binary classification program that takes in the adult data set and is able to assess whether or not a given adult in the set makes more than $50k/yr as measured by the accuracy metric. **

## Solution Statement

As noted, this dataset performs very well with linear models. As such we will be developing our suite of linear models to develop our program.  In the classification context, this means we will be using logistic regression.  We might also try a support vector classifier with a linear kernel.  

## Benchmark Model

As a benchmark model, we will simply guess the most dominant class.

## Performance Metric

We will be using Accuracy since this is the default metric for a classification problem. `#TODO: Why did we pick this?`

## Project Roadmap

1. Gather and store data
1. Establish sampling procedure
    - https://www.surveymonkey.com/mp/sample-size-calculator/
    - Don't always have to pull the whole thing, even when you have the space.
    - Use 90% confidence interval, 1% margin of error on 48842 instances - 5911 as a representative sample. Maybe take 3 samples and look at the mean
    - We might think about using bootstrapping or repetitive sampling in some places
1. EDA - data cleaning
    - Fill the nan/missing values
    - Check the dtypes
1. EDA - statistics
   1. EDA - descriptive
   1. EDA - correlation and distribution - features
   1. EDA - correlation and distribution - target
1. EDA - handle categorical features
1. Establish performance metric 
   - confusion matrix
   - accuracy
   - precision
   - f-score
1. Benchmark Model
1. Standardize Data
1. Skew-Normalize & Standardize Data
1. Investigate outliers
   - box plot
   - joint plot
1. Bias-Variance Tradeoff in sample size
1. Principal Component Analysis
1. Examine Scree Plot
1. Segmentation
1. Feature selection
1. Develop model pipelines
1. Gridsearch model pipelines

## Stretch Goals

1. Explore individual feature relevance
  - Apply `f_classif` 
1. Reduce the feature set
1. Explore other data projections:
   - Polynomial expansions