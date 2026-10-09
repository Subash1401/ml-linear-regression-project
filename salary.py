
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data = {
    "Age": [22, 25, None, 28, 30, 35, 40, 45, 50, 27, 24, 26],

    "Experience": [0, 2, 3, 5, 7, 10, 15, 20, 25, 4, 2, 3],

    "City": [
        "Chennai",
        "Chennai",
        "Bangalore",
        None,
        "Chennai",
        "Mumbai",
        "Chennai",
        "Bangalore",
        "Chennai",
        "Chennai",
        "Mumbai",
        "Chennai"
    ],

    "Education": [
        "BE",
        "BE",
        "BTech",
        "ME",
        "BE",
        "MBA",
        "BE",
        "BTech",
        "PhD",
        "BE",
        "phD",
        "MBA"
    ],

    "Salary": [
        25000,
        30000,
        35000,
        40000,
        45000,
        60000,
        80000,
        100000,
        500000,
        42000,
        32000,
        35000
    ]
}


df = pd.DataFrame(data)

print("Original Data:")
print(df)



print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nData Information:")
print(df.info())

print("\nStatistical Information:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())



df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)




df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)


print("\nAfter Handling Missing Values:")
print(df)



print("\nNumber of Duplicates:")
print(df.duplicated().sum())

df = df.drop_duplicates()



Q1 = df["Salary"].quantile(0.25)

Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1


lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR


print("\nQ1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Bound:", lower)

print("Upper Bound:", upper)


# Find outliers

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]

print("\nOutliers:")
print(outliers)


# Remove outliers

df = df[
    (df["Salary"] >= lower) &
    (df["Salary"] <= upper)
]


print("\nData After Removing Outliers:")
print(df)




# Convert categorical columns into numbers

df = pd.get_dummies(
    df,
    columns=["City", "Education"],
    dtype=int
)


print("\nAfter Encoding:")
print(df)



# X = Input features
# y = Target

X = df.drop("Salary", axis=1)

y = df["Salary"]


print("\nX:")
print(X)

print("\ny:")
print(y)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)




scaler = StandardScaler()


# Learn scaling from training data
X_train_scaled = scaler.fit_transform(X_train)


# Apply same scaling to test data
X_test_scaled = scaler.transform(X_test)


print("\nScaled Training Data:")
print(X_train_scaled)


model = LinearRegression()



model.fit(
    X_train_scaled,
    y_train
)


print("\nModel Training Completed!")



y_pred = model.predict(
    X_test_scaled
)


print("\nActual Salary:")
print(y_test.values)

print("\nPredicted Salary:")
print(y_pred)


mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==========================")
print("MODEL EVALUATION")
print("==========================")

print("MAE:", mae)

print("R2 Score:", r2)