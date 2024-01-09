#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from scipy.stats import kurtosis, skew
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.linalg import svd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
import warnings
from keras.models import Sequential
from keras.layers import Dense


# In[54]:


data= pd.read_csv('Titanic-Dataset.csv')


# In[8]:


data.head(5)


# In[9]:


num_rows, num_columns = data.shape
print("number of rows:",num_rows)
print("number of columns:",num_columns)


# In[10]:


data.info()


# In[11]:


data.describe()


# In[12]:


data.isna().any(axis=0) #check missing values


# In[13]:


data.isnull().sum() #count of missing or NAN values


# In[14]:


# Remove rows with NaN values in any row
data.dropna(axis=0, inplace=True)
data.isna().any(axis=0)


# In[15]:


# Remove rows with NaN values in any column
data.dropna(axis=1, inplace=True)
data.isna().any(axis=1)


# In[16]:


#fill missing data with unknown
data['Age'].fillna('Unknown', inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)
data['Embarked'].fillna('Unknown', inplace=True)

data.isnull().sum() #shows the count of missing values after filling


# In[17]:


data.head(20)


# In[18]:


data.shape


# In[19]:


data.hist(figsize=(10,10))


# In[20]:


#plot a bar plot for the survival of passengers
survived_counts = data['Survived'].value_counts()


plt.bar(survived_counts.index.astype(str), survived_counts.values, color=['#8B0000', '#4169E1'])
plt.xlabel('Survived')
plt.ylabel('Number of Passengers')
plt.title('Survival Distribution')
plt.show()


# In[21]:


#define numerical cols 
numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch']
numerical_data = data[numerical_columns]

#fill the missing values with their column mean
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())


# **covariance matrix**

# In[22]:


# Display column names to check for existence
print(data.columns)
#assign defined cols to x 
x = data[numerical_columns]

#Computes the covariance matrix for the selected numerical columns 
covariance_matrix = np.cov(x, rowvar=False)
print(covariance_matrix)

plt.figure(figsize=(6, 5))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=numerical_columns, yticklabels=numerical_columns)
plt.title('Covariance Matrix ')
plt.show()


# covariance measures the degree to which two random variables change together. It indicates the direction of the linear relationship between two variables and the strength between them

# **correlation**

# In[23]:


correlation_matrix = numerical_data.corr()


print(correlation_matrix)


plt.figure(figsize=(6, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=numerical_columns, yticklabels=numerical_columns)
plt.title('Correlation Matrix ')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


# the colors in the heatmap indicate the strength and direction of these relationships, making it easier to identify patterns or dependencies between different features or variables in the dataset.

# **chi-square**

# In[24]:


contingency_table_titanic = pd.crosstab(data['Survived'], data['Pclass'])
#tabulates the frequency distribution of observations for 'Survived' and 'Pclass'
#using pandas and the crosstab data
chi2_stat_titanic, p_value_titanic, dof_titanic, expected_titanic = chi2_contingency(contingency_table_titanic)


print("Chi-square Test:")
print("Chi2 Statistic:", chi2_stat_titanic)
print("P-value:", p_value_titanic)
print("Degrees of Freedom:", dof_titanic)
print("Expected Frequencies Table:")
print(expected_titanic)


# In[25]:


contingency_table_titanic = pd.crosstab(data['Survived'], data['Pclass'])

# Perform the chi-square test
chi2_stat_titanic, p_value_titanic, dof_titanic, expected_titanic = chi2_contingency(contingency_table_titanic)

print("Chi-square Test:")
print("Chi2 Statistic:", chi2_stat_titanic)
print("P-value:", p_value_titanic)
print("Degrees of Freedom:", dof_titanic)
print("Expected Frequencies Table:")
print(expected_titanic)

# Plotting observed vs expected frequencies
categories = ['Pclass 1', 'Pclass 2', 'Pclass 3']
bar_width = 0.35
index = range(len(categories))

plt.bar(index, contingency_table_titanic.iloc[0], bar_width, label='Observed: Survived = 0')
plt.bar(index, expected_titanic[0], bar_width, label='Expected: Survived = 0', alpha=0.5)

plt.bar([i + bar_width for i in index], contingency_table_titanic.iloc[1], bar_width, label='Observed: Survived = 1')
plt.bar([i + bar_width for i in index], expected_titanic[1], bar_width, label='Expected: Survived = 1', alpha=0.5)

plt.xlabel('Pclass')
plt.ylabel('Frequency')
plt.title('Observed vs Expected Frequencies by Pclass and Survival')
plt.xticks([i + bar_width / 2 for i in index], categories)
plt.legend()
plt.tight_layout()
plt.show()


# **T-test**

# In[26]:


numerical_col1 = 'Age'
numerical_col2 = 'Fare'


data = data.dropna(subset=[numerical_col1, numerical_col2])


values1 = data[numerical_col1]
values2 = data[numerical_col2]


t_test_result = ttest_ind(values1, values2)
print("t-test statistic:", t_test_result.statistic)
print("P-value:", t_test_result.pvalue)


# **ANOVA**

# In[27]:


age_by_pclass = []
for pclass in sorted(data['Pclass'].unique()):
    age_by_pclass.append(data[data['Pclass'] == pclass]['Age'].dropna())

# Performing one-way ANOVA
anova_result_titanic = f_oneway(*age_by_pclass)

# Displaying the results
print("ANOVA:")
print("F-statistic:", anova_result_titanic.statistic)
print("p-value:", anova_result_titanic.pvalue)


# **PCA & LDA**

# In[28]:


# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
data[['Age', 'Fare']] = imputer.fit_transform(data[['Age', 'Fare']])

data = data.dropna()  # Drop rows with any remaining missing values

# Define features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Encoding categorical variables using .loc to avoid SettingWithCopyWarning
encoder = LabelEncoder()
X.loc[:, 'Sex'] = encoder.fit_transform(X['Sex'])
X.loc[:, 'Embarked'] = encoder.fit_transform(X['Embarked'])

# Feature scaling for PCA (LDA doesn't require scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply LDA with reduced components
lda = LinearDiscriminantAnalysis(n_components=1)  # Adjusted to 1 component
X_lda = lda.fit_transform(X_scaled, y)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('PCA of Titanic Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], [0] * len(X_lda), c=y, cmap='viridis', edgecolor='k')  # Only one dimension for LDA
plt.title('LDA of Titanic Dataset')
plt.xlabel('Linear Discriminant 1')
plt.yticks([])
plt.tight_layout()

plt.show()


# **SVD**

# In[29]:


# Ensure the DataFrame is correctly named (use 'titanic_data' as loaded from the CSV)
data = data.copy()

# Check and drop columns if they exist
columns_to_drop = ['Name', 'Ticket', 'Cabin']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(columns=col)

# Proceed with label encoding
label_encoders = {}
for column in ['Sex', 'Embarked']:
    label_encoders[column] = LabelEncoder()
    data[column] = data[column].fillna('Missing')  # Fill missing values
    data[column] = label_encoders[column].fit_transform(data[column])

# Handle missing values in other columns
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Now the data is ready for SVD
from scipy.linalg import svd
U, S, Vt = svd(data, full_matrices=False)

# Select the number of components to keep
n_components = 5
reduced_data = U[:, :n_components] * S[:n_components]

# Create a reduced DataFrame
reduced_df = pd.DataFrame(reduced_data, columns=[f'Component_{i}' for i in range(n_components)])

print("Shapes of SVD components:", U.shape, S.shape, Vt.shape)
print("Reduced data:\n", reduced_df.head())



# **Naive Bayesian**

# In[30]:


print("Columns before dropping:", data.columns)

# Drop the columns if they exist in the DataFrame
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Display the columns after dropping to verify
print("Columns after dropping:", data.columns)

# Drop rows with missing values
data = data.dropna() 


# In[31]:


X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()  
y = data['Survived']

# Encoding categorical variables using .loc to avoid SettingWithCopyWarning
encoder = LabelEncoder()
X.loc[:, 'Sex'] = encoder.fit_transform(X['Sex'])
X.loc[:, 'Embarked'] = encoder.fit_transform(X['Embarked'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Naive Bayes classifier
naive_bayes = GaussianNB()

# Fit the model on the training data
naive_bayes.fit(X_train, y_train)

# Predict on the test data
y_pred = naive_bayes.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# **Bayesian Belief Network (BBN)**

# In[40]:


data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Handling missing values
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Replace missing ages with the mean age
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Replace missing embarked with the mode

# Binning the age into groups
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 18, 65, 100], labels=['Child', 'Adult', 'Senior'])

# Calculate probabilities manually
# P(Survived | Pclass, Sex)
survived_given_class_sex = data.groupby(['Pclass', 'Sex', 'Survived']).size().unstack().fillna(0)
survived_given_class_sex = survived_given_class_sex.div(survived_given_class_sex.sum(axis=1), axis=0)

# P(Survived | AgeGroup)
survived_given_age = data.groupby(['AgeGroup', 'Survived']).size().unstack().fillna(0)
survived_given_age = survived_given_age.div(survived_given_age.sum(axis=1), axis=0)

# Querying the probabilities
# P(Survived = 1 | Pclass = 1, Sex = 0)
prob_survived_given_class_sex = survived_given_class_sex.loc[(1, 0), 1]
print(f"Probability of Survival given Pclass=1 and Sex=0 (female): {prob_survived_given_class_sex}")

# P(Survived | AgeGroup = 'Adult')
prob_survived_given_age_adult = survived_given_age.loc['Adult']
print(f"Probability of Survival for Adults: \n{prob_survived_given_age_adult}")


# In[41]:


# Encoding categorical variables
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Define features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']


# **Decision tree**

# In[42]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree using Entropy as criterion
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")


# **K-NN (different distances)** & **data splitting**

# In[43]:


# Assuming X and y are defined previously
warnings.filterwarnings("ignore")

# Impute missing values in X
imputer = SimpleImputer(strategy='mean')  # Use an appropriate imputation strategy
X_imputed = imputer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Define K-NN classifiers with different distance metrics
k_values = [3, 5, 7]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

for k in k_values:
    for metric in distance_metrics:
        # Train K-NN classifier
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)

        # Predict on the test set
        y_pred = knn.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"K = {k}, Distance Metric = {metric}: Accuracy = {accuracy:.4f}")


# **k-fold cross validation and average accuracy**

# In[44]:


# Initialize KFold and Decision Tree Classifier
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = DecisionTreeClassifier(random_state=42)

fold = 1
accuracies = []
# Perform K-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Fold {fold} Accuracy: {accuracy}")
    fold += 1

# Calculate average accuracy
avg_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy: {avg_accuracy}")


# **confusion matrix**

# In[45]:


# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot evaluation metrics
metrics = ['Accuracy', 'Error Rate', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, error_rate, precision, recall, f1]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['blue', 'red', 'green', 'orange', 'purple'])
plt.title('Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)


# In[46]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Now checking for overfitting or underfitting
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

if train_accuracy > test_accuracy:
    print("The model might be overfitting.")
elif train_accuracy < test_accuracy:
    print("The model might be underfitting.")
else:
    print("The model has a balanced fit.")


# **Neural Networks**

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (if not already done)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


# **ROC**

# In[55]:


data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Handling missing values
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Define features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data['Survived']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
knn = KNeighborsClassifier(n_neighbors=3)

# Cross-validation and ROC curve calculation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(knn, X_train, y_train, cv=cv)
fpr, tpr, _ = roc_curve(y_train, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




