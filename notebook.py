# Test data cleaning
df_train = pd.read_csv('https://raw.github.com/mattdelhey/kaggle-titanic/master/Data/train.csv', index_col=False)
# fill null values with median fo rage column....
df_train["age"] = df_train["age"].fillna(df_train["age"].median())

# fill null values with median for embarked column....
df_train["embarked"] = df_train["embarked"].fillna(df_train["embarked"].mode()[0])

# drop columns .......
df_train.drop(['cabin'], axis=1, inplace = True)

df_train.drop(['name','ticket'],axis=1, inplace=True)

# convert age as int
df_train['age'] = df_train['age'].astype(int)

# one hot encode Embarked column and drop it from the original df
encoded_df = pd.get_dummies(df_train['embarked'], prefix='embarked')
df_encoded = pd.concat([df_train, encoded_df], axis=1)
df_encoded = df_encoded.drop('embarked', axis=1)

# one hot encode Class column and drop it from the original df
encoded_df = pd.get_dummies(df_encoded['pclass'], prefix='pclass')
df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
df_encoded = df_encoded.drop('pclass', axis=1)

# one hot encode Sex column and drop it from the original df
encoded_df = pd.get_dummies(df_encoded['sex'], prefix='sex')
df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
df_encoded = df_encoded.drop('sex', axis=1)

# Splitting the data into features and target variable
X = df_encoded.drop('survived', axis=1)  # Features
y = df_encoded['survived']  # Target variable

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
