import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def remove_zero_presence_rows(df):
    filtered_df = df[df['Presence'] != 0].copy()
    return filtered_df

df = pd.read_csv(r'C:\Users\SFATESS\Documents\Sisard\GPU\predictions.csv')
df.dropna(inplace=True)
df = remove_zero_presence_rows(df)

X = df[['discriminator']]
y = df['Presence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()
gradient_boosting = GradientBoostingClassifier()
svm_classifier = SVC(probability=True)
naive_bayes = GaussianNB()

stacking_model = StackingClassifier(estimators=[
    ('logistic_regression', logistic_regression),
    ('random_forest', random_forest),
    ('gradient_boosting', gradient_boosting),
    ('svm_classifier', svm_classifier),
    ('naive_bayes', naive_bayes)
], final_estimator=LogisticRegression())

stacking_model.fit(X_train_resampled, y_train_resampled)

predictions = stacking_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print(f"\nAccuracy on Test Set: {accuracy}")
print("Classification Report:")
print(report)

conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = set(y)
tick_marks = [i for i in range(len(classes))]
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), ha="center", va="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.show()

probabilities = stacking_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

def calculate_confusion_matrix_values(y_true, y_pred, df):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    tp_values = df.loc[(y_true == 1) & (y_pred == 1), 'Presence'].values
    fp_values = df.loc[(y_true == 0) & (y_pred == 1), 'Presence'].values
    tn_values = df.loc[(y_true == 0) & (y_pred == 0), 'Presence'].values
    fn_values = df.loc[(y_true == 1) & (y_pred == 0), 'Presence'].values

    return tp_values, fp_values, tn_values, fn_values

tp_values, fp_values, tn_values, fn_values = calculate_confusion_matrix_values(y_test, predictions, df)

print(f"True Positives Values: {tp_values}")
print(f"False Positives Values: {fp_values}")
print(f"True Negatives Values: {tn_values}")
print(f"False Negatives Values: {fn_values}")
