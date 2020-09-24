# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:28:49 2019

@author: CINO
"""

#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data set
data_set=pd.read_csv("TitanicSurvival.csv")

#split IDV and DV
x=data_set.iloc[:,2:5].values
y=data_set.iloc[:,1:2].values


#deal with categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
x[:,2]=labelencoder_x.fit_transform(x[:,2])
onehotencoder_x=OneHotEncoder(categorical_features=[0,2])

x=onehotencoder_x.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y[:,0]=labelencoder_x.fit_transform(y[:,0])
onehotencoder_y=OneHotEncoder(categorical_features=[0])
y=onehotencoder_y.fit_transform(y).toarray()

num_first_class_suvive=0
num_second_class_suvive=0
num_third_class_suvive=0
for each_row in range(0,1309):
    if y[each_row,1]==1:
        if x[each_row,2]==1:
            num_first_class_suvive=num_first_class_suvive+1
        elif x[each_row,3]==1:
            num_second_class_suvive=num_second_class_suvive+1
        elif x[each_row,4]==1:
            num_third_class_suvive=num_third_class_suvive+1    

######################################################################################
######################################################################################
######################################################################################
#extract basic information
            
each_age_group_survive_list=[0]*8
for each_row in range(0,1309):
    if x[each_row,5]<=10:
        each_age_group_survive_list[0]=each_age_group_survive_list[0]+1
    elif x[each_row,5]>10 and x[each_row,5]<=20 and y[each_row,1]==1:
        each_age_group_survive_list[1]=each_age_group_survive_list[1]+1
    elif x[each_row,5]>20 and x[each_row,5]<=30 and y[each_row,1]==1:
        each_age_group_survive_list[2]=each_age_group_survive_list[2]+1
    elif x[each_row,5]>30 and x[each_row,5]<=40 and y[each_row,1]==1:
        each_age_group_survive_list[3]=each_age_group_survive_list[3]+1
    elif x[each_row,5]>40 and x[each_row,5]<=50 and y[each_row,1]==1:
        each_age_group_survive_list[4]=each_age_group_survive_list[4]+1
    elif x[each_row,5]>50 and x[each_row,5]<=60 and y[each_row,1]==1:
        each_age_group_survive_list[5]=each_age_group_survive_list[5]+1
    elif x[each_row,5]>60 and x[each_row,5]<=70 and y[each_row,1]==1:
        each_age_group_survive_list[6]=each_age_group_survive_list[6]+1
    elif x[each_row,5]>70 and x[each_row,5]<=80 and y[each_row,1]==1:
        each_age_group_survive_list[7]=each_age_group_survive_list[7]+1


total_sex_list=[]
total_age_list=[]
total_class_list=[]
total_survial_list=[]

for each_row in range(0,1309):
    total_sex_list.append(x[each_row,1])
    total_age_list.append(x[each_row,5])
    if x[each_row,2]==1:
        total_class_list.append(1)
    if x[each_row,3]==1:
        total_class_list.append(2)
    if x[each_row,4]==1:
        total_class_list.append(3)
    total_survial_list.append(y[each_row,1])

######################################################################################
#visual raw data

# 1.ratio of men to women 
num_female=0
for each in x[:,0]:
    if each==1:
        num_female=num_female+1
        
num_male=0
for each in x[:,1]:
    if each==1:
        num_male=num_male+1
sectors='female','male'
sizes=[num_female,num_male]
colors=['pink','blue']
plt.figure(1)
plt.title('Ratio of male over female')
plt.pie(sizes,labels=sectors,colors=colors,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.show()

# 2.ratio of survived and dead
num_dead=0
num_survive=0
for each in y[:,0]:
    if each==1:
        num_dead=num_dead+1
    if each==0:
        num_survive=num_survive+1

sectors='dead','survive'
sizes=[num_dead,num_survive]
colors=['red','green']
plt.figure(2)
plt.title('Ratio of dead over survive')
plt.pie(sizes,labels=sectors,colors=colors,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.show()  

# 3. ratio of classes
num_first_class=0
num_second_class=0
num_third_class=0

for each in x[:,2]:
    if each==1:
        num_first_class=num_first_class+1
for each in x[:,3]:
    if each==1:
        num_second_class=num_second_class+1
for each in x[:,4]:
    if each==1:
        num_third_class=num_third_class+1

sectors='first class','second class','third class'
sizes=[num_first_class,num_second_class,num_third_class]

plt.figure(3)
plt.title('Ratio of each classes')
plt.pie(sizes,labels=sectors,autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.show()
 

# 4. age vs survival
age_list=[]
for each in x[:,5]:
    age_list.append(each)
survival_list=[]
for each in y[:,1]:
    survival_list.append(each) 


# 5.age range
plt.figure(4)
plt.title("Age range")
plt.xlabel("age interval")
plt.ylabel("number of observation")
plt.hist(age_list,range=(0,80))
plt.show()

# 6. sex vs survival
num_dead_female=0
num_suvive_female=0
num_dead_male=0
num_survive_male=0
for each_row in range(0,1309):
    if y[each_row,1]==1 and x[each_row,0]==1:
        num_suvive_female=num_suvive_female+1
    elif y[each_row,1]==0 and x[each_row,0]==1:
        num_dead_female=num_dead_female+1
    elif y[each_row,1]==1 and x[each_row,1]==1:
        num_survive_male=num_survive_male+1
    elif y[each_row,1]==0 and x[each_row,1]==1:
        num_dead_male=num_dead_male+1
        
        
plt.figure(5)
plt.title("Sex vs Survival(1)")
plt.xlabel("survive=0 dead=1")
plt.ylabel("female=0 male=1")
x=[0,0,1,1]
y=[0,1,1,0]
plt.scatter(x[0],y[0],s=num_suvive_female*10,color="pink")
plt.text(x[0],y[0],"female survive")
plt.scatter(x[1],y[1],s=num_survive_male*10,color="blue")
plt.text(x[1],y[1],"male survive")
plt.scatter(x[2],y[2],s=num_dead_male*10,color="blue")
plt.text(x[2],y[2],"male dead")
plt.scatter(x[3],y[3],s=num_dead_female*10,color="pink")
plt.text(x[3],y[3],"female dead")
plt.show()

plt.figure(6)
plt.title("Sex vs Survival(2)")
survive_to_total_ratio_sex_list=[num_suvive_female/num_female,num_survive_male/num_male]
x_axis=[1,2]
text_list=["female","male"]
plt.bar(x_axis,survive_to_total_ratio_sex_list,color="green")
plt.xticks(x_axis,text_list)
plt.ylabel("percent of survival")
plt.xlabel("sex")
plt.show()

# 7.age vs survival
plt.figure(7)
plt.title("Age vs Survival")
plt.xlabel("age group")
x=[0]*8
for each_age in age_list:
    if each_age<=10:
        x[0]=x[0]+1
    elif each_age>10 and each_age<=20:
        x[1]=x[1]+1
    elif each_age>20 and each_age<=30:
        x[2]=x[2]+1
    elif each_age>30 and each_age<=40:
        x[3]=x[3]+1
    elif each_age>40 and each_age<=50:
        x[4]=x[4]+1
    elif each_age>50 and each_age<=60:
        x[5]=x[5]+1
    elif each_age>60 and each_age<=70:
        x[6]=x[6]+1
    elif each_age>70 and each_age<=80:
        x[7]=x[7]+1
x_axis=[1,2,3,4,5,6,7,8]
plt.ylabel("percent of survival")
text_list=["0 to 10","10 to 20","20 to 30","30 to 40","40 to 50","50 to 60","60 to 70" ,"70 to 80"]
survive_to_total_ratio_age_list=[]
for each_group in range(0,len(x)):
    value=each_age_group_survive_list[each_group]/x[each_group]
    survive_to_total_ratio_age_list.append(value)
plt.bar(x_axis,survive_to_total_ratio_age_list,color="green") 
plt.xticks(x_axis,text_list,fontsize=10,rotation=30) 
plt.show()  

# 8. class vs survival
plt.figure(8)
plt.title("Class vs Survival")
survive_to_total_ratio_class_list=[num_first_class_suvive/num_first_class,
                                   num_second_class_suvive/num_second_class,
                                   num_third_class_suvive/num_third_class]
x_axis=[1,2,3]
text_list=["first class","second class","third class"]
plt.bar(x_axis,survive_to_total_ratio_class_list,color="green")
plt.ylabel("percent of survival")
plt.xticks(x_axis,text_list)
plt.xlabel("class")
plt.show()

# 9. all in one
from mpl_toolkits import mplot3d
plt.figure(9)
ax=plt.axes(projection='3d')

for each_row in range(0,1309):
    if total_survial_list[each_row]==1:
        ax.scatter3D(total_age_list[each_row],total_class_list[each_row],total_sex_list[each_row],
                     color="green")
    elif total_survial_list[each_row]==0:
        ax.scatter3D(total_age_list[each_row],total_class_list[each_row],total_sex_list[each_row],
                     color="red")
plt.xlabel("age")
plt.ylabel("class")
ax.set_zlabel("sex (1=male 0=female)")

y_axis=[1,2,3]
y_text=["1st class","2nd class","3rd class"]
plt.yticks(y_axis,y_text)
plt.show()

######################################################################################
######################################################################################
######################################################################################
#survival prediction

import tensorflow as tf

'''
three factors x:
1. sex (male=0, female=1)
2. age
3. class

y=0: survive
y=1: dead
'''
#data preprocessing
x=data_set.iloc[:,2:5].values.tolist()
y=data_set.iloc[:,1:2].values.tolist()

x_train=[]
y_train=[]

for i in range(0,len(x)):
    x_value=[0,0,0]
    y_value=0
    
    #sex
    if x[i][0]=='female':
       x_value[0]=1
    
    #age
    x_value[1]=x[i][1]
    
    #class
    if x[i][2]=='2nd':
        x_value[2]=1
    elif x[i][2]=='3rd':
        x_value[2]=2
    
    #survive?
    if y[i][0]=='no':
        y_value=1    
    x_train.append(x_value)
    y_train.append(y_value)

x_train=np.array(x_train)
y_train=np.array(y_train)

#split train and test
index_list=np.arange(0,len(x))
np.random.shuffle(index_list)
x_train=x_train[index_list,:]
y_train=y_train[index_list]

num_train=int(len(x)*0.8)
x_test=np.copy(x_train[num_train:,:])
y_test=np.copy(y_train[num_train:])

x_train=x_train[:num_train,:]
y_train=y_train[:num_train]

#using single layer perceptron
dtype=tf.float64
input_node=tf.placeholder(shape=(None,3),dtype=dtype)
weight=tf.Variable([0,0,0],dtype=dtype)
bias=tf.Variable([0],dtype=dtype)
output_node=tf.squeeze(tf.sigmoid(tf.matmul(input_node,weight[:,tf.newaxis])+bias))
'''
output_node=tf.greater(output_node,0.5)
output_node=tf.cast(output_node, dtype)
'''
#before learning
sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
output=sess.run(output_node,feed_dict={input_node:x_train})

#training
mse_train=tf.compat.v1.losses.mean_squared_error(output_node,y_train)
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(mse_train)

print("before train : ",sess.run(mse_train,feed_dict={input_node:x_train}))

num_iteration=10000
mse_train_list=[]
for i in range(0,num_iteration):
    sess.run(train,feed_dict={input_node:x_train})
    mse=sess.run(mse_train,feed_dict={input_node:x_train})
    mse_train_list.append(mse)
    if i%100==0:
        print(mse)
        
plt.plot(np.arange(0,len(mse_train_list)),mse_train_list)

#training set prediction
y_pred=sess.run(output_node,feed_dict={input_node:x_train})
y_pred_train=np.copy(y_pred)
y_pred_train[y_pred>0.5]=1
y_pred_train[y_pred<=0.5]=0

acc=y_pred_train==y_train
acc_train=np.zeros(acc.shape)
acc_train[acc==True]=1
acc_train=sum(acc_train)/len(acc_train)
print("acc_train : ",acc_train)

#test set prediction
y_pred=sess.run(output_node,feed_dict={input_node:x_test})
y_pred_test=np.copy(y_pred)
y_pred_test[y_pred>0.5]=1
y_pred_test[y_pred<=0.5]=0

acc=y_pred_test==y_test
acc_test=np.zeros(acc.shape)
acc_test[acc==True]=1
acc_test=sum(acc_test)/len(acc_test)
print("acc_test : ",acc_test)

sess.close()
######################################################################################
######################################################################################
######################################################################################
#using single layer perceptron
dtype=tf.float64
input_node=tf.placeholder(shape=(None,3),dtype=dtype)
w1=tf.Variable([[0,0,0],
               [0,0,0],
               [0,0,0]],dtype=dtype)
b1=tf.Variable([0,0,0],dtype=dtype)
hidden1=tf.nn.relu(tf.matmul(input_node,w1)+b1)
w2=tf.Variable([0,0,0],dtype=dtype)
b2=tf.Variable([0],dtype=dtype)
output_node=tf.squeeze(tf.sigmoid(tf.matmul(hidden1,w2[:,tf.newaxis])+b2))

'''
output_node=tf.greater(output_node,0.5)
output_node=tf.cast(output_node, dtype)
'''
#before learning
sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
output=sess.run(output_node,feed_dict={input_node:x_train})

#training
mse_train=tf.compat.v1.losses.mean_squared_error(output_node,y_train)
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(mse_train)

print("before train : ",sess.run(mse_train,feed_dict={input_node:x_train}))

num_iteration=10000
mse_train_list=[]
for i in range(0,num_iteration):
    sess.run(train,feed_dict={input_node:x_train})
    mse=sess.run(mse_train,feed_dict={input_node:x_train})
    mse_train_list.append(mse)
    if i%100==0:
        print(mse)
        
plt.plot(np.arange(0,len(mse_train_list)),mse_train_list)

#training set prediction
y_pred=sess.run(output_node,feed_dict={input_node:x_train})
y_pred_train=np.copy(y_pred)
y_pred_train[y_pred>0.5]=1
y_pred_train[y_pred<=0.5]=0

acc=y_pred_train==y_train
acc_train=np.zeros(acc.shape)
acc_train[acc==True]=1
acc_train=sum(acc_train)/len(acc_train)
print("acc_train : ",acc_train)

#test set prediction
y_pred=sess.run(output_node,feed_dict={input_node:x_test})
y_pred_test=np.copy(y_pred)
y_pred_test[y_pred>0.5]=1
y_pred_test[y_pred<=0.5]=0

acc=y_pred_test==y_test
acc_test=np.zeros(acc.shape)
acc_test[acc==True]=1
acc_test=sum(acc_test)/len(acc_test)
print("acc_test : ",acc_test)