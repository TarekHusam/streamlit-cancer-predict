import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Clean and Normlize
def get_clean_data():

    #Read data
    data = pd.read_csv('data.csv')

    # Drop the ID number
    data = data.drop(['id'], axis=1)
    # Drop the Unnamed: 32 column
    data = data.drop(['Unnamed: 32'], axis=1)
    
    # Encode the diagnosis variable
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data

#Create the Model - Train & Test 
def create_model(data):
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    #Scale the data (Normlize)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    #Split the data
    x_train,x_test,y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=42)

    #Train the data
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #Test the data
    y_pred = model.predict(x_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))



    return model, scaler



#Main 
def main():
    data = get_clean_data()

    model , scaler =create_model(data)

#For storing the Data for later
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    


if __name__ == '__main__':
    main()