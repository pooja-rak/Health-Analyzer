import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
#dataset 
df=pd.read_csv('generated_height_weight_status_dataset.csv')
print(df)
label=LabelEncoder()
y=label.fit_transform(df['status'])
x=df[['age','height','weight']]
model=KNeighborsClassifier(n_neighbors=25)
model.fit(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=56)
y_pred=model.predict(x_test)
ar=accuracy_score(y_test, y_pred)
print(f"The Accuracy of the Model: {ar*100:.2f}")
con=confusion_matrix(y_test, y_pred)
print("\n The Confusion Matrix of the Given Dataset \n", con)
cr=classification_report(y_test, y_pred)
print("The Classification Report of the Model\n",cr)
sns.heatmap(con, annot=True,fmt='d',cmap='Greens')
plt.show()

joblib.dump(model,'weight.pkl')
joblib.dump(label,'label.pkl')

