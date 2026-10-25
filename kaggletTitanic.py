import sqlite3
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

path_0 = r"C:\Projects\FOLDER 2\PY2\New folder\test.csv"
path_1 = r"C:\Projects\FOLDER 2\PY2\New folder\train.csv"
path_2 = r"C:\Projects\FOLDER 2\PY2\New folder\titanicK.db"

def createDB(path_0,path_1,path_2):
    df = pd.read_csv(path_0)
    df_1 = pd.read_csv(path_1)

    df["Sex"] = np.where(df["Sex"] == "female",1,0)
    df_1["Sex"] = np.where(df_1["Sex"] == "female",1,0)


    df.fillna({"Age":int(df["Age"].mean())}, inplace=True)
    df.fillna( {"Fare":int(df["Fare"].mean())}, inplace=True)
    df_1.fillna({"Age":int(df_1["Age"].mean())}, inplace=True)

    conn = sqlite3.connect(path_2)

    df.to_sql("test", conn, if_exists="replace", index=False)
    df_1.to_sql("train", conn, if_exists="replace",index=False)
def train(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT Pclass, Sex, Age, Fare, Survived, CASE " \
    "WHEN Sex = 1 AND Age < 25 THEN 1 ELSE 0 END AS youngFemale FROM train")

    rows = cursor.fetchall()
    X_train = np.array([[row[0], row[1], row[2], row[3], row[5]] for row in rows], dtype=float)
    Y_train = np.array([row[4] for row in rows], dtype=int)

    print("x shape", X_train.shape)
    print("bincount", np.bincount(Y_train))

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, random_state=42)

    model = RandomForestClassifier(class_weight= 'balanced', n_estimators = 100, random_state= 40)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))
    print(model.feature_importances_)

    cursor.execute("SELECT PassengerId, Pclass, Sex, Age, Fare, CASE "
        "WHEN Sex = 1 AND Age < 25 THEN 1 ELSE 0 END AS youngFemale FROM test")
    test_rows = cursor.fetchall()

    test_ids = [row[0] for row in test_rows]
    X_test_kaggle = np.array([row[1:6] for row in test_rows],dtype=float)
    kaggle_predictions = model.predict(X_test_kaggle)

    with open(r'C:\Projects\FOLDER 2\PY2\submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PassengerId', 'Survived'])
        for a, b in zip(test_ids, kaggle_predictions):
            writer.writerow([a,b])


    conn.commit()
    conn.close()

def main():
    createDB(path_0,path_1,path_2)
    train(path_2)

if __name__ == "__main__":
    main()
