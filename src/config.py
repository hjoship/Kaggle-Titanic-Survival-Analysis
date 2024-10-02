# Feature columns for model training
FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Categorical and numerical columns
CAT_COLS = ['Sex', 'Embarked']
NUM_COLS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Default columns to display
DEFAULT_DISPLAY_COLUMNS = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare']

# Model parameters
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Test size for train-test split
TEST_SIZE = 0.2
