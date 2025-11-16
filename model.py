import os
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import logging
#logger = logging.getLogger('model')
from log import setup_logging
logger = setup_logging('model')


class ModelDevelopment:
    def __init__(self):

        self.OUTPUT_DIR = r"C:\Users\golis\OneDrive\Desktop\crps\output"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        print(f"Output directory verified: {self.OUTPUT_DIR}")
        logger.info(f" Output directory verified: {self.OUTPUT_DIR}")

    def common(self, X_train, y_train, X_test, y_test):
        #Train, evaluate multiple models, select best by AUC, and return the best model
        try:
            classifiers = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
                "Random Forest": RandomForestClassifier(
                    criterion='entropy', n_estimators=100, random_state=42
                ),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "SVM": SVC(kernel='rbf', probability=True),
                "Gradient Boosting": GradientBoostingClassifier()
            }

            plt.figure(figsize=(8, 6))
            best_auc, best_model, best_name = 0, None, ""

            for name, model in classifiers.items():
                logger.info(f"🔹 Training {name}...")
                model.fit(X_train, y_train)

                # Save each model
                model_path = os.path.join(self.OUTPUT_DIR, f"{name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                y_pred = model.predict(X_test)
                logger.info(f"------ {name} ------")
                logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                # AUC calculation
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                except Exception:
                    y_prob = model.decision_function(X_test)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                logger.info(f"AUC Score ({name}): {roc_auc:.4f}")
                plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

                # Track best model
                if roc_auc > best_auc:
                    best_auc, best_model, best_name = roc_auc, model, name

            # Plot ROC curves
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves - All Models")
            plt.legend(loc="lower right")
            #  Save the ROC plot to your output folder
            plt.savefig(os.path.join(self.OUTPUT_DIR, "roc_comparison.png"))
            plt.show()



            logger.info(f"##Best Model: {best_name} | AUC: {best_auc:.4f}")

            # Save best model as model.pkl
            best_model_path = os.path.join(self.OUTPUT_DIR, "model.pkl")
            with open(best_model_path, "wb") as f:
                pickle.dump(best_model, f)
            logger.info(f"## Best model saved to: {best_model_path}")

            return best_model

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Issue at line {er_lin.tb_lineno}: {er_msg}")
            return None
