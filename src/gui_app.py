from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import webbrowser
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from PIL import Image, ImageTk
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#declaring Variables
global filename
global X,Y
global dataset
global main
global text
global accuracy, precision, recall, fscore
global X_train, X_test, y_train, y_test
global classifier
global label_encoder, labels, columns, types, pca
#designing main screen
main = tkinter.Tk()
main.title("Detection Of Network Layer Attacks In Wireless Sensor Networks") 
main.geometry("1920x1020")
# setting and re-sizing background image 
# Load and store the image globally
image = Image.open("Background_image.PNG")
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(copy_of_image)

# Create label and store reference
label = tkinter.Label(main, image=photo)
label.image = photo  # prevent garbage collection
label.pack(fill=BOTH, expand=YES)

# Handle window resizing
def resize_image(event):
    new_width = event.width
    new_height = event.height
    resized = copy_of_image.resize((new_width, new_height))
    new_photo = ImageTk.PhotoImage(resized)
    label.config(image=new_photo)
    label.image = new_photo

label.bind('<Configure>', resize_image)

# Call this after Tk() is initialized
def getLabel(name):
    label = -1
    for i in range(len(labels)):
        if name == labels[i]:
            label = i
            break
    return label
#connecting backend to the screen button
def screen():
    print("Start Detection button clicked")
    inButton.destroy()
    OutputScreen()
def OutputScreen():
    print("OutputScreen running")
    global text
    font1 = ('times', 14, 'bold')
    text=Text(main,height=20,width=80)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=60,y=250)
    text.config(font=font1)
  # uploading the dataset
    def uploadDataset():
        global filename, dataset, labels
        text.delete('1.0', END)
        filename = filedialog.askdirectory(initialdir=".")
        text.insert(END,filename+" loaded\n\n")
        df1 = pd.read_csv(filename+"/sybil_DNS.csv")
        df2 = pd.read_csv(filename+"/sybil_LDAP.csv")
        df3 = pd.read_csv(filename+"/wormhole_MSSQL.csv")
        df4 = pd.read_csv(filename+"/wormhole_NTP.csv")
        df5 = pd.read_csv(filename+"/wormhole_NetBIOS.csv")
        df6 = pd.read_csv(filename+"/sybil_SNMP.csv")
        df7 = pd.read_csv(filename+"/wormhole_SSDP.csv")
        df8 = pd.read_csv(filename+"/sybil_UDP.csv")
        df9 = pd.read_csv(filename+"/sybil_Syn.csv")
        df10 = pd.read_csv(filename+"/wormhole_UDP_LAG.csv")
        dataset = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]
        dataset = pd.concat(dataset)
        labels = np.unique(dataset['Label']).tolist()
        print(labels)
        text.insert(END,str(dataset.head()))
        text.update_idletasks()
        attack = dataset.groupby('Label').size()
        attack.plot(kind="bar")
        plt.xlabel('Attacks')
        plt.ylabel('Number of Records')
        plt.title('Different Attacks found in dataset')
        plt.show() # plotting the attacks in the dataset with records
# preprocessing the dataset
    def preprocessDataset():
        global dataset, label_encoder, X, Y, columns, types, pca
        global X_train, X_test, y_train, y_test
        text.delete('1.0', END)
        label_encoder = []
        columns = dataset.columns
        types = dataset.dtypes.values
        for i in range(len(types)):
            name = types[i]
            if name == 'object' and columns[i] != 'Label':
                le = LabelEncoder()
                dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
                label_encoder.append(le)
                print(columns[i])
        dataset.fillna(0, inplace = True)
        Y = dataset['Label'].ravel()
        temp = []
        for i in range(len(Y)):
            temp.append(getLabel(Y[i]))
        temp = np.asarray(temp)
        Y = temp
        dataset = dataset.values
        X = dataset[:,0:dataset.shape[1]-1]
        X = normalize(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        print(np.unique(Y))
        text.insert(END,"Dataset after features processing & normalization\n\n")
        text.insert(END,str(X)+"\n\n")
        text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
        text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
        pca = PCA(n_components = 50)
        X = pca.fit_transform(X)
        # splitting the dataset into test and train datasets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        text.insert(END,"Dataset Train and Test Split\n\n")
        text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
        text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    # calculating metrics to the all algorithms
    def calculateMetrics(algorithm, predict, y_test):
        a = accuracy_score(y_test,predict)*100
        p = precision_score(y_test, predict,average='macro') * 100
        r = recall_score(y_test, predict,average='macro') * 100
        f = f1_score(y_test, predict,average='macro') * 100
        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
        text.insert(END,algorithm+" Precision : "+str(p)+"\n")
        text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
        text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
        text.update_idletasks()
        print(np.unique(predict))
        print(np.unique(y_test))
        conf_matrix = confusion_matrix(y_test, predict)
        #plt.figure(figsize =(6, 6))
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title(algorithm+" Confusion matrix") 
        plt.ylabel('True class') 
        plt.xlabel('Predicted class') 
        plt.show()
    # training the XGBoost algorithm
    def runXGBoost():
        global classifier
        global X, Y, X_train, X_test, y_train, y_test
        global accuracy, precision,recall, fscore
        accuracy = []
        precision = []
        recall = []
        fscore = []
        text.delete('1.0', END)
        if os.path.exists('model/xgb.txt'):
            with open('model/xgb.txt', 'rb') as file:
                xgb_cls = pickle.load(file)
            file.close()        
        else:
            xgb_cls = XGBClassifier()
            xgb_cls.fit(X_train, y_train)
            with open('model/xgb.txt', 'wb') as file:
                pickle.dump(xgb_cls, file)
                file.close() 
        predict = xgb_cls.predict(X_test)
        classifier=xgb_cls
        calculateMetrics("XGBoost", predict, y_test)
            # training the KNN algorithm
    def runKNN():
        if os.path.exists('model/knn.txt'):
            with open('model/knn.txt', 'rb') as file:
                knn_cls = pickle.load(file)
            file.close()        
        else:
            knn_cls = KNeighborsClassifier(n_neighbors = 2) 
            knn_cls.fit(X_train, y_train)
            with open('model/knn.txt', 'wb') as file:
                pickle.dump(knn_cls, file)
            file.close() 
        predict = knn_cls.predict(X_test)
        calculateMetrics("KNN", predict, y_test)
            
    # Training the AdaBoost Algorithm
    def runAdaBoost():
        if os.path.exists('model/adb.txt'):
            with open('model/adb.txt', 'rb') as file:
                adb_cls = pickle.load(file)
                file.close()        
        else:
            adb_cls = AdaBoostClassifier()
            adb_cls.fit(X_train, y_train)
            with open('model/adb.txt', 'wb') as file:
                pickle.dump(adb_cls, file)
            file.close()        
        predict = adb_cls.predict(X_test)
        calculateMetrics("AdaBoost", predict, y_test)
    # Training the Random Forest Algorithm
    def runRandomForest():
        global classifier
        if os.path.exists('model/rf.txt'):
            with open('model/rf.txt', 'rb') as file:
                rf = pickle.load(file)
            file.close()        
        else:
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            with open('model/rf.txt', 'wb') as file:
                pickle.dump(rf, file)
            file.close()        
        predict = rf.predict(X_test)
        classifier = rf
        calculateMetrics("Random Forest", predict, y_test)
        
        
    # Prediction of attacks in the test dataset
    def predict():
        global label_encoder, labels, columns, types, pca, classifier, text
        text.delete('1.0', END)
        filename = filedialog.askopenfilename(initialdir="testData")
        testData = pd.read_csv(filename)
        count = 0
        for i in range(len(types)-1):
            name = types[i]
            if name == 'object':
                if columns[i] == 'Flow Bytes/s':
                    testData[columns[i]] = pd.Series(label_encoder[count].fit_transform(testData[columns[i]].astype(str)))
                else:
                    testData[columns[i]] = pd.Series(label_encoder[count].fit_transform(testData[columns[i]].astype(str)))
                count += 1           
        testData.fillna(0, inplace=True)
        testData = testData.values
        testData = normalize(testData)
        testData = pca.transform(testData)
        predictions = classifier.predict(testData)  # this is now a NumPy array

        for i in range(len(predictions)):
            text.insert(END, f"Test DATA : {str(testData[i])} ===> PREDICTED AS {labels[predictions[i]]}\n\n")

    
# graphs for the metrics of all algorithms which means confusion matrix
    def graph():
        output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
        output+="<th>FSCORE</th></tr>"
        output+="<tr><td>XGBoost Algorithm</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
        output+="<tr><td>AdaBoostBoost Algorithm</td><td>"+str(accuracy[2])+"</td><td>"+str(precision[2])+"</td><td>"+str(recall[2])+"</td><td>"+str(fscore[2])+"</td></tr>"
        output+="<tr><td>KNN Algorithm</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
        output+="<tr><td>Random Forest Algorithm</td><td>"+str(accuracy[3])+"</td><td>"+str(precision[3])+"</td><td>"+str(recall[3])+"</td><td>"+str(fscore[3])+"</td></tr>"
        output+="</table></body></html>"
        f = open("table.html", "w")
        f.write(output)
        f.close()
        webbrowser.open("table.html",new=2)
        df = pd.DataFrame([['XGBoost','Precision',precision[0]],['XGBoost','Recall',recall[0]],['XGBoost','F1 Score',fscore[0]],['XGBoost','Accuracy',accuracy[0]],['KNN','Precision',precision[1]],['KNN','Recall',recall[1]],['KNN','F1 Score',fscore[1]],['KNN','Accuracy',accuracy[1]],
                           ['AdaBoost','Precision',precision[2]],['AdaBoost','Recall',recall[2]],['AdaBoost','F1 Score',fscore[2]],['AdaBoost','Accuracy',accuracy[2]],
                           ['Random Forest','Precision',precision[3]],['Random Forest','Recall',recall[3]],['Random Forest','F1 Score',fscore[3]],['Random Forest','Accuracy',accuracy[3]],
                          ],columns=['Algorithms','Performance Output','Value'])
        df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
        plt.show()
    font1 = ('times', 18, 'bold')
    uploadButton = Button(main, text="Upload  Dataset", command= lambda:[OutputScreen(),uploadDataset()])
    uploadButton.place(x=900,y=250)
    uploadButton.config(font=font1)

    preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
    preprocessButton.place(x=1200,y=250)
    preprocessButton.config(font=font1) 

    xgButton = Button(main, text="Run XGBoost Algorithm", command=runXGBoost)
    xgButton.place(x=900,y=350)
    xgButton.config(font=font1)

    knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
    knnButton.place(x=1200,y=350)
    knnButton.config(font=font1)

    adaboostButton = Button(main, text="Run AdaBoost Algorithm", command=runAdaBoost)
    adaboostButton.place(x=900,y=450)
    adaboostButton.config(font=font1)

    rfButton = Button(main, text="Run RandomForest Algorithm", command=runRandomForest)
    rfButton.place(x=1200,y=450)
    rfButton.config(font=font1)

    predictButton = Button(main, text="Predict Attack from Test Data", command=predict)
    predictButton.place(x=900,y=550)
    predictButton.config(font=font1)

 
font = ('times', 20, 'bold')
title = Label(main, text='DETECTION OF NETWORK LAYER ATTACKS IN WIRELESS SENSOR NETWORK')
title.config(bg= 'gray', fg='black')  
title.config(font=font)           
title.config(height=3, width=100)       
title.place(x=0,y=3)
font1 = ('times', 24, 'bold')
inButton = Button(main, text="Start Detection", command= screen)
inButton.place(x=650,y=450)
inButton.config(font=font1)
main.mainloop()










