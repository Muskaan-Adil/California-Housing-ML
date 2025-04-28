### Detailed Report

---

#### 1. **Basic Data Inspection**:
To begin, I loaded the California Housing Prices dataset and performed an initial inspection.  
I printed the first five rows using `head()` to get a quick overview and used `info()` to understand the data types and identify any missing values.  
Additionally, the `describe()` method gave a statistical summary, which helped me understand the distribution of numerical features like `total_rooms`, `population`, `median_income`, and `median_house_value`.

---

#### 2. **Missing Values**:
Next, I checked for missing data using `isnull().sum()`.  
The output showed that the `total_bedrooms` column had missing values.  
Since missing values can negatively impact model performance, I handled them by filling the missing entries with the **median** value of the `total_bedrooms` column.  
This method ensured the data remained unbiased by outliers.

---

#### 3. **Handling Categorical Variables**:
The dataset included a categorical feature: `ocean_proximity`.  
Since the focus of this project was building a **simple linear regression model with one numerical feature**, I dropped the `ocean_proximity` column entirely.  
For future, more complex models, this feature could be encoded rather than dropped.

---

#### 4. **Feature Selection**:
After cleaning, I selected the **independent variable** (`median_income`) and the **dependent variable** (`median_house_value`).  
The choice of `median_income` was based on a preliminary correlation analysis that showed it had the strongest relationship with the target variable.

---

#### 5. **Feature Scaling**:
I standardized the `median_income` feature using **StandardScaler** from `sklearn.preprocessing`.  
Although scaling is not mandatory for simple linear regression, it is a good practice to ensure that features have a mean of 0 and a standard deviation of 1, especially when preparing for future model improvements.

---

#### 6. **Train-Test Split**:
I divided the data into a training set (80%) and a testing set (20%) using `train_test_split` from `sklearn.model_selection`.  
This separation allows for unbiased evaluation of model performance on unseen data.

---

#### 7. **Model Training**:
Using `LinearRegression` from `sklearn.linear_model`, I trained the model on the scaled training data.  
The model learned the relationship between `median_income` and `median_house_value`.

---

#### 8. **Prediction and Evaluation**:
I made predictions on the test set and evaluated the model's performance using three key metrics:
- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, which brings the error metric back to the original units.
- **R-squared Score (R²)**: Indicates the proportion of the variance in the target variable explained by the model.

The results were:
- **MSE:** _Displayed from script output_
- **RMSE:** _Displayed from script output_
- **R² Score:** _Displayed from script output_

The model showed a strong positive relationship, meaning **higher median incomes tend to correspond to higher median house values**.

---

#### 9. **Visualization**:
To visually assess the model, I plotted:
- A **scatter plot** of the actual `median_income` vs. `median_house_value`.
- The **regression line** to show the model’s predictions.

The scatter plot confirmed a clear positive trend, although some spread around the regression line indicated natural variability not captured by a single feature.

---

### Final Thoughts:
This project successfully demonstrated the use of a **simple linear regression model** to predict house prices based on income levels.  
Key strengths included careful data cleaning, proper feature selection, and thoughtful evaluation.  

While the model captures the general trend well, housing prices are influenced by many other factors.  
For future improvements:
- Incorporate multiple features (e.g., `housing_median_age`, `population`).
- Explore categorical feature encoding.
- Experiment with more advanced models like **Polynomial Regression** or **Tree-based models** for potentially better accuracy.