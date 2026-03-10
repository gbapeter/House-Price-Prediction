# Overview
This project applies Machine Learning techniques to predict house prices using property features such as number of bedrooms, bathrooms, square footage, and location details. The model is trained on housing data and learns the patterns that influence real estate prices.
The goal of the project is to demonstrate how supervised machine learning models can be used for regression problems, specifically predicting housing prices.

# Technologies Used
- Python: 
- Pandas:
- NumPy:
- Scikit-learn:
- Joblib:
- Streamlit:
- Jupyter Notebook:
- Github: 

# Project Structure
## Dataset Description
The dataset contains housing information and property attributes used to predict the target variable (price).
Each row represents a house listing.
### Columns in the Dataset
- date: Date the house was listed or sold
- price: Price of the house (Target Variable)
- Bedrooms: Number of bedrooms in the house
- Bathrooms: Number of bathrooms
- sqft_living: Square footage of the living area
- sqft_lot: Total land area of the property
- floors: Number of floors in the house
- waterfront: Indicates if the property has a waterfront view
- view: Quality of the view
- condition: Condition rating of the house
- sqft_above: Square footage above ground level
- sqft_basement: Square footage of the basement
- yr_built: Year the house was built
- yr_renovated: Year the house was renovated
- street: Street address
- city: City where the property is located
- statezip: State and ZIP code
- country: Country of the property

## Machine Learning Workflow
The project followed a standard machine learning pipeline:
1. Data Loading
The dataset was loaded using Pandas for analysis and preprocessing.

2. Exploratory Data Analysis (EDA)
- Checked the datatypes of each column.
- Checked the summary statistics of the numerical columns.
- Checked for outliers and distribution using a histogram and box-plot of all numerical columns. Outliers were detected but not removed because they are expected to be in a housing dataset.
- Checked for missing values (No missing values were detected). 
- Checked duplicates (no duplicates were found).
- Checked the city column for typographical errors and inconsistencies (no issues were found).
- Checked rows where bedroom, bathroom and price were 0.
- Checked the number of times each city occurred.
- Checked the relationship between numerical variables using a heatmap and correlation

3. Feature Engineering
New columns were created to help the model learn better. They include: year_sold, month, age, years_since_renovation, renovate (to show whether house was renovated or not), has_basement, and zipcode.

4. Data Cleaning
Cleaning steps include:
- Removing unnecessary columns:
	- The date column was dropped after using it to extract the month and year.
	- The yr_sold, yr_renovated, statezip and yr_built columns were dropped because years do little to help a model detect patterns.
	- The street column was dropped because a model can not easily pick patterns from it and it could cause issues during modelling.
	- The country column was dropped because it had only one value (USA). The model wouldn’t have picked any pattern from it.
	- The sqft_above and sqft_basement columns were dropped because they added up to give the sqft_living column, hence, they were a repetition. Also, sqft_living has been known to influence house prices more.
- Converting data types
	- The city column was converted to numbers using OneHotEncoder because RandomForest and LinearRegression can not handle textual data.
	- The date column was converted to datetime data type.
- The price (target) column was log transformed because it was right-skewed and had outliers.
- Removed 0s in price because a house should not cost $0. There were 49 rows (1% of dataset) with that issue.
- Further investigation revealed that only two rows had 0s for bathroom and bedroom columns and both had multiple floors. This suggested wrong entries and should be removed. It is possible that houses built specifically for commercial purpose would not have bedrooms and bathrooms, but since that could not be known for this dataset, it felt safer to drop them, especially as only two rows were affected.
 
6. Data Splitting
The dataset was divided into:
X (Features) → independent variables
y (Target) → house price
Example:
y = df['price']
X = df.drop(columns=['price'])

Then the dataset was split into training and testing set.

6. Model Training
A machine learning regression model was trained on the dataset to learn patterns that affect house prices. Linear Regression and Random Forest Regressor were used.

7. Model Evaluation
The performance of the model was evaluated using regression metrics such as: Mean Absolute Error (MAE), Root Mean Squared Error (MSE) and R² Score.


# Contributors
- Gba Peter Ternenge
- Amina Isa
- Nicholas Happiness

