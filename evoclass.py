import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score, matthews_corrcoef, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from dateutil import parser
import re
import time


class MyModel:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.rf_model = RandomForestClassifier(criterion='gini', 
                                n_estimators=100, #1
                                max_features=None, 
                                min_samples_split=2,#2
                                min_samples_leaf=10,#1
                                max_depth=10)#10

    def prepare_data(self, data):
        validActivationDate = []
        ccDiagnose = []
        cancerMessageValidity = []

        date_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"  # Regular expression pattern for the desired date format
        # i.e., 2023-01-30T21:50:20

        for index, row in data.iterrows():
            valid_activation = 1
            diagnose_date = 1
            start_date = 1

            activation_date_str = str(row['activationDate'])
            diagnose_date_str = str(row['cancerCase.diagnosedato'])
            opp_start_date_str = str(row['cancerCase.datoOppstartStralebeh'])
            #prosedyredato_str =str(row['cancerCase.prosedyredato'])
            
            if not re.match(date_pattern, activation_date_str):
                valid_activation = 0

            # Checking for cancer case validity in terms of date format.    
            
            if not re.match(date_pattern, diagnose_date_str) or not re.match(date_pattern, opp_start_date_str):
                diagnose_date = 0

            validActivationDate.append(valid_activation)
            ccDiagnose.append(diagnose_date)

            messages = row['cancerMessages']
            is_valid = 1 
            if isinstance(messages, str):
                message_list = json.loads(str(messages).replace("'", "\""))
                for message in message_list:
                    message_diagnose_date_str = str(message.get('diagnosedato', ''))
                    message_dodsdato_str = str(message.get('dodsdato', ''))
                    message_meldingsdato_str = str(message.get('meldingsdato', ''))
                    message_fodselsdato_str = str(message.get('fodselsdato', ''))

            
                    if not re.match(date_pattern, message_diagnose_date_str) or not re.match(date_pattern, message_dodsdato_str) or not re.match(date_pattern, message_meldingsdato_str) or not re.match(date_pattern, message_fodselsdato_str):
                        is_valid = 0
                        break

            cancerMessageValidity.append(is_valid)
            

        # Create new features/columns with extracted values
        data['validActivationDate'] = validActivationDate
        data['ccDiagnose'] = ccDiagnose
        data['is_no_auth'] = (data['Auth'] == 'NoAuth').astype(int)
        data['cancerMessagesNr'] = data['cancerMessages'].apply(lambda row: len(json.loads(str(row).replace("'", "\""))) if isinstance(row, str) else 0)

        data['cancerMessageValidity'] = cancerMessageValidity

        cancerTypesNr = []
        for types in data['cancerTypes']:
            if isinstance(types, str):
                types_list = types.split(',')
                cancerTypesNr.append(len(types_list))
            else:
                cancerTypesNr.append(0)

        data['cancerTypesNr'] = cancerTypesNr
        
        # Drop the original features/columns
        data = data.drop(['activationDate', 'cancerCase.diagnosedato', 'cancerCase.datoOppstartStralebeh', 'Auth', 'cancerMessages', 'cancerTypes'], axis=1)

        X = data.drop(['Status Code'], axis=1)
        y = data['Status Code'].apply(lambda x: 1 if x == 200 else 0) # Convert to binary classification

        X.to_csv('prepare_encoded_data.csv', index=False)

        # X_new = X.drop(list_keys, axis=1)
        # X_new.to_csv('prepare_encoded_data_clean.csv', index=False)
        features = X.columns

        return X, y, features

    def fit_encoder(self, X, features):
        for feature in features:
            self.encoder[feature].fit(X[feature])

    def prepare_encoded_data(self, X, features):
        X_encoded = X.copy()

        for feature in features:
            X_encoded[feature] = self.encoder[feature].transform(X_encoded[feature])

        X_encoded.to_csv('prepare_encoded_data.csv', index=False)   
        return X_encoded

    def train(self, data, random_seed, run):
        X, y, features = self.prepare_data(data)
        self.encoder = {}
        for feature in features:
            self.encoder[feature] = LabelEncoder()
        self.fit_encoder(X, features)
        X_encoded = self.prepare_encoded_data(X, features)
        print(X_encoded.shape[1])

# 'API Url','Individual nr', 'cancerMessagesNr', 'cancerCase.topografiICD10'
        # X_encoded = X_encoded.drop(elements, axis=1)

        # scores = cross_val_score(self.rf_model, X_encoded, y, cv=5)
        # print(scores)
        # return X_encoded, y


        # X, y, features = model.train(data)

        # print(X_encoded)
        # print(features)
        # print(y)
        

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=500, train_size=1000, random_state=random_seed, stratify=y)
        # X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, train_size=0.8, random_state=random_seed, stratify=y)
        print(len(X_train))
        print(len(X_test))

        self.rf_model.fit(X_train, y_train)
        n_params = sum(tree.tree_.node_count for tree in self.rf_model.estimators_) * 5
        print(f"Total number of parameters: {n_params:d}")


        feature_importances = self.rf_model.feature_importances_

        feature_names = X_train.columns

        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        feature_importance_df.to_csv('features_importance'+str(run)+'.csv', index=False)

        print(feature_importance_df)

        y_pred = self.rf_model.predict(X_test)

        model_info = {
            'encoder': self.encoder,
            'rf_model': self.rf_model,
            'features': features,
        }

        # Save the model information
        filename = 'model'+str(run)+'.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(model_info, file)

        # Calculate accuracy score, classification report, and AUC-ROC score
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, self.rf_model.predict_proba(X_test)[:, 1])
        auc_prc = average_precision_score(y_test, self.rf_model.predict_proba(X_test)[:, 1], pos_label=1)
        rf_mcc = matthews_corrcoef(y_test, y_pred)

        # Generate the classification report and AUC-ROC score
        print("Classification report:\n", report)
        print("Accuracy: ", accuracy)
        print("AUC-ROC score:", auc_roc)
        print("AUC-PRC score:", auc_prc)
        print("MCC score:", rf_mcc)
        return X_encoded, y, features, accuracy

        # # Plot ROC curve
        # fpr, tpr, thresholds = roc_curve(y_test, self.rf_model.predict_proba(X_test)[:, 1])
        # plt.plot(fpr, tpr, color='green', label='Random Forest (AUC = %0.2f)' % auc_roc)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.show()
        #
        # # Plot Precision-Recall curve
        # precision, recall, thresholds = precision_recall_curve(y_test, self.rf_model.predict_proba(X_test)[:, 1])
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.show()
        #
        # cm = confusion_matrix(y_test, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
        # disp.plot()
        # plt.title('Confusion Matrix')
        # plt.show()

    def prepare_new_data(self, action_data, individual_index, action_index, http_request_info):
        print(action_data, individual_index, action_index, http_request_info)
        df = pd.json_normalize(action_data)

        df = pd.DataFrame(df)
       
        encoder = self.encoder  
        features = self.features.tolist()

        X_encoded = df.copy()
        # X_encoded.to_csv('single-data.csv', index=False)        

        http_request_parts = http_request_info.split(",")
        api_url_value = http_request_parts[0].strip()
        auth_value = http_request_parts[1].split("auth=")[-1].strip()
        action_type = http_request_parts[0].split(" ")[0].strip()
        api_url = http_request_parts[0].split(" ")[1].strip()
        X_encoded['API Url'] = api_url
        #X_encoded['Action type'] = action_type
        date_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"  

        for column in features:
            if column in X_encoded.columns:
                X_encoded[column] = X_encoded[column].map(lambda x: encoder[column].transform([x])[0] if x in encoder[column].classes_ else -1)
          
        missing_columns = set(features) - set(X_encoded.columns)
        for column in missing_columns:
            X_encoded[column] = 0

        valid_activation = 1
        valid_diagnose = 1
        try:
            activation_date_str = df['activationDate'].iloc[0]
            if not re.match(date_pattern, activation_date_str):
                valid_activation = 0
        except:
            pass
        try:
            # Check if the 'cancerCase' column exists
            if 'cancerCase' in df.columns:
                if not df['cancerCase'].empty:
                    try:
                        diagnose_date_str = df['cancerCase'].iloc[0]['diagnosedato']
                        opp_start_date_str = df['cancerCase'].iloc[0]['datoOppstartStralebeh']
                        if not re.match(date_pattern, diagnose_date_str) or not re.match(date_pattern, opp_start_date_str):
                            valid_diagnose = 0
                    except:
                        pass
        except:
            pass        

        cancerTypesNr = []
        if 'cancerTypes' in df.columns:
            for types in df['cancerTypes']:
                if isinstance(types, list):
                    cancerTypesNr.append(len(types))
                else:
                    cancerTypesNr.append(0) 
        else:
                    cancerTypesNr.append(0) 
        
        
        cancerMessageValidity = 1 
        if 'cancerMessages' in df.columns:
            for messages in df['cancerMessages']:
                if isinstance(messages, list):
                    
                    message_list = json.loads(str(messages).replace("'", "\""))
                    for message in message_list:
                        message_diagnose_date_str = str(message.get('diagnosedato', ''))
                        message_dodsdato_str = str(message.get('dodsdato', ''))
                        message_meldingsdato_str = str(message.get('meldingsdato', ''))
                        message_fodselsdato_str = str(message.get('fodselsdato', ''))

                        if not re.match(date_pattern, message_diagnose_date_str) or not re.match(date_pattern, message_dodsdato_str) or not re.match(date_pattern, message_meldingsdato_str) or not re.match(date_pattern, message_fodselsdato_str):
                            cancerMessageValidity = 0
                            break
     
        X_encoded['Individual nr'] = individual_index
        X_encoded['Action nr'] = action_index
        X_encoded['is_no_auth'] = int(auth_value == 'NoAuth')

        # df['is_no_auth'] = (df['Auth'] == 'NoAuth').astype(int)

        X_encoded['cancerMessagesNr'] = 0
        if 'cancerMessages' in df.columns:
            try:
                X_encoded['cancerMessagesNr'] = df['cancerMessages'].apply(lambda row: len(json.loads(str(row).replace("'", "\""))) if isinstance(row, str) else len(row) if isinstance(row, list) else 0)
            except:
                pass  

        X_encoded['validActivationDate'] = valid_activation
        X_encoded['ccDiagnose'] = valid_diagnose
        X_encoded['cancerTypesNr'] = cancerTypesNr
        # X_encoded['is_no_auth'] = 0
        X_encoded['cancerMessageValidity'] = cancerMessageValidity

        X_encoded = X_encoded[features]
        X_encoded.to_csv('single.csv', index=False)   
        return X_encoded 
        
           

    def predict_new_data(self, X_encoded_new):
        print("X_encoded_new", X_encoded_new)
        # Make predictions on the new data
        start_time = time.time()
        y_pred_new = self.rf_model.predict(X_encoded_new)
        end_time = time.time()
        prediction_time_ms = (end_time - start_time) * 1000

        print("p", y_pred_new)
        # Return the predictions
        return y_pred_new, prediction_time_ms


# Load data into a Pandas dataframe
data = pd.read_csv('dataset.csv')

model = MyModel()
accuracy_list = []
accuracy_lists = []
for run in range(10):
    X, y, feature, accuracy = model.train(data, 2, run)
    accuracy_lists.append(accuracy)
print(sum(accuracy_lists)/len(accuracy_lists))