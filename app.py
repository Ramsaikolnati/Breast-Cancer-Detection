from flask import Flask,render_template,request,session
import mysql.connector
import numpy as np
import pickle

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier



app=Flask(__name__)
app.secret_key = "abc"  

con=mysql.connector.connect(database='cancer',user='root',password='')
cur=con.cursor()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/profile")
def profile():
    uid=session['id']
    s="select * from users where id="+str(uid)
    cur.execute(s)
    data=cur.fetchall()
    return render_template("profile.html",d=data)


@app.route("/adminlogin")
def adminlogin():
    return render_template("adminlogin.html")


@app.route("/userlogin")
def userlogin():
    return render_template("userlogin.html")

@app.route("/nb")
def nb():
    global nbacc
    dataset = shuffle(np.array(pd.read_csv("dataset.csv",header=1)))
    data_frame = pd.read_csv("dataset.csv",header=1)
    data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
    dataset = shuffle(np.array(data_frame))

    extracted_dataset= []
    target = []

    #extract target column
    for row in dataset:
        extracted_dataset.append(row[1:])
        target.append(row[0])


    X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)


    model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    model.fit(X_train,Y_train)
    filename = 'nbmodel.pkl'
    pickle.dump(model, open(filename, 'wb'))
    predicted = model.predict(X_test)

    nbacc=accuracy_score(Y_test, predicted)
        
    return render_template("nb.html",nbac=(nbacc*100))

@app.route("/dt")
def dt():
    global dtacc
    dataset = shuffle(np.array(pd.read_csv("dataset.csv",header=1)))
    data_frame = pd.read_csv("dataset.csv",header=1)
    data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
    dataset = shuffle(np.array(data_frame))

    extracted_dataset= []
    target = []

    #extract target column
    for row in dataset:
        extracted_dataset.append(row[1:])
        target.append(row[0])


    X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 50)
    clf_entropy .fit(X_train,Y_train)
    predicted = clf_entropy .predict(X_test)
    filename = 'dtmodel.pkl'
    pickle.dump(clf_entropy, open(filename, 'wb'))
    
    dtacc=accuracy_score(Y_test, predicted)
        
    return render_template("dt.html",dtac=(dtacc*100))

@app.route("/knn")
def knn():
    global knnacc
    dataset = shuffle(np.array(pd.read_csv("dataset.csv",header=1)))
    data_frame = pd.read_csv("dataset.csv",header=1)
    data_frame.drop(data_frame.columns[[0]], axis=1, inplace=True)
    dataset = shuffle(np.array(data_frame))

    extracted_dataset= []
    target = []

    #extract target column
    for row in dataset:
        extracted_dataset.append(row[1:])
        target.append(row[0])


    X_train, X_test, Y_train, Y_test= train_test_split(extracted_dataset,target,test_size=0.3)
    clf = KNeighborsClassifier(n_neighbors=5,algorithm='brute',p=1)
    clf.fit(X_train,Y_train)


    predicted = clf.predict(X_test)

    filename = 'knnmodel.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
    knnacc=accuracy_score(Y_test, predicted)
        
    return render_template("knn.html",knnac=(knnacc*100))



@app.route("/compare")
def compare():
    import matplotlib.pyplot as plt

    #data
    x = [1, 2, 3]
    h = [nbacc, dtacc,knnacc]
    c = ['red', 'yellow', 'orange']

    #bar plot
    plt.bar(x, height = h, color = c)

    plt.savefig('static/abc.jpg')

    return render_template("compare.html")


@app.route("/signup")
def signup():
    return render_template("signup.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/prediction",methods=['POST'])
def prediction():
    a1=request.form['t1']
    a2=request.form['t2']
    a3=request.form['t3']
    a4=request.form['t4']
    a5=request.form['t5']
    a6=request.form['t6']

    a7=request.form['t7']
    a8=request.form['t8']
    a9=request.form['t9']
    a10=request.form['t10']
    
    a11=request.form['t11']
    a12=request.form['t12']
    a13=request.form['t13']
    a14=request.form['t14']
    a15=request.form['t15']
    a16=request.form['t16']

    a17=request.form['t17']
    a18=request.form['t18']
    a19=request.form['t19']
    a20=request.form['t20']

    
    a21=request.form['t21']
    a22=request.form['t22']
    a23=request.form['t23']
    a24=request.form['t24']
    a25=request.form['t25']
    a26=request.form['t26']

    a27=request.form['t27']
    a28=request.form['t28']
    a29=request.form['t29']
    a30=request.form['t30']
    newdata=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27,a28,a29,a30]

    filename = 'dtmodel.pkl'
    model = pickle.load(open(filename, 'rb'))
    fres=model.predict([newdata])
    return render_template("prediction.html",fr=fres)


@app.route("/viewdataset")
def viewdataset():
    from csv import reader
    lst=[]

    with open("dataset.csv","r", encoding = 'unicode_escape') as obj:
        csv_reader=reader(obj)
        lst=list(csv_reader)
    return render_template("viewdataset.html",data=lst)

@app.route("/adminloginDB",methods=['POST'])
def adminloginDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    if uname=='admin' and pwd=='diet':
        return render_template("adminhome.html")
    else:
        return render_template("adminlogin.html",msg="Pls Check the Credentials")

@app.route("/userloginDB",methods=['POST'])
def userloginDB():
    uname=request.form['un']
    pwd=request.form['pwd']
    s="select * from users where email='"+uname+"' and password='"+pwd+"'"
    cur.execute(s)
    data=cur.fetchall()
    
    if len(data)>0:
        session['id']=data[0][0] 
        return render_template("userhome.html")
    else:
        return render_template("userlogin.html",msg="Pls Check the Credentials")



@app.route("/userRegDB",methods=['POST'])
def userRegDB():
    name=request.form['name']
    mail=request.form['mail']
    contact=request.form['contact']
    add=request.form['add']
    pwd=request.form['pwd']

    s="insert into users(name,email,contact,address,password) values('"+name+"','"+mail+"','"+contact+"','"+add+"','"+pwd+"')"
    
    cur.execute(s)
    con.commit()
    return render_template("userlogin.html")


@app.route("/updateProfile",methods=['POST'])
def updateProfile():
    uid=session['id']
    name=request.form['name']
    mail=request.form['mail']
    contact=request.form['contact']
    add=request.form['add']
    pwd=request.form['pwd']

    s="update users set email='"+mail+"',contact='"+contact+"',address='"+add+"',password='"+pwd+"' where id='"+str(uid)+"'"
    print(s)
    cur.execute(s)
    con.commit()
    return render_template("userhome.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/ViewFeedback")
def ViewFeedback():
    cur.execute("SELECT name,email,contact,feedback FROM feedback,users WHERE feedback.uid=users.id")
    data=cur.fetchall()
    return render_template("ViewFeedback.html",d=data)


@app.route("/feedbackDB",methods=['POST'])
def feedbackDB():
    uid=session['id']
    fb=request.form['fb']
    s="insert into feedback(feedback,uid) values('"+fb+"','"+str(uid)+"')"
    cur.execute(s)
    con.commit()
    return render_template("feedbacksubmit.html")


@app.route("/logout")
def logout():
    return render_template("index.html")

app.run(debug=True)
