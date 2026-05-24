APP_TITLE = "Automated Data Modeling Tools"

DEFAULT_MAX_ROWS = 20000
DEFAULT_MAX_FEATURES = 50
DEFAULT_CV_FOLDS = 3
DEFAULT_TPE_TRIALS = 30

MAX_ROWS_CAP = 50000
MAX_FEATURES_CAP = 200
MAX_CV_CAP = 5
MAX_TPE_TRIALS_CAP = 100

CLASSIFICATION_MODELS = [
	"Logistic Regression",
	"Decision Tree",
	"Random Forest",
	"Gradient Boosting",
	"SVM",
	"KNN",
]

REGRESSION_MODELS = [
	"Linear Regression",
	"Ridge",
	"Lasso",
	"Decision Tree",
	"Random Forest",
	"Gradient Boosting",
	"SVR",
	"KNN",
]

STATISTICAL_CLASSIFICATION_MODELS = ["Logistic Regression"]
STATISTICAL_REGRESSION_MODELS = ["Linear Regression", "Ridge", "Lasso"]
ML_CLASSIFICATION_MODELS = [
	"Decision Tree",
	"Random Forest",
	"Gradient Boosting",
	"SVM",
	"KNN",
]
ML_REGRESSION_MODELS = [
	"Decision Tree",
	"Random Forest",
	"Gradient Boosting",
	"SVR",
	"KNN",
]

FEATURE_SELECTION_METHODS = [
	"None",
	"Stepwise",
	"RFE",
	"SelectKBest",
	"Model-based",
]

STEPWISE_DIRECTIONS = ["forward", "backward", "bidirectional"]

MODEL_BASED_OPTIONS = ["L1/Lasso", "Tree importance"]

TUNING_METHODS = ["None", "Grid Search", "Manual", "TPE (Optuna)"]

ENABLE_LLM_SYSTEM_MESSAGES = False

CLASSIFICATION_METRICS = {
	"Accuracy": "accuracy",
	"F1 (weighted)": "f1_weighted",
	"ROC-AUC (ovr)": "roc_auc_ovr",
}

REGRESSION_METRICS = {
	"RMSE": "neg_root_mean_squared_error",
	"MAE": "neg_mean_absolute_error",
	"R2": "r2",
}
