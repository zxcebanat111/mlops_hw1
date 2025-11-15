from catboost import CatBoostClassifier
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Define optimal threshold
model_th = 0.98
logger.info('Pretrained model imported successfully...')


# Make predictions
def make_pred(dt, path_to_file):

    # Make submission dataframe
    predictions = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': model.predict_proba(dt)[:, 1],
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return predictions


# Make submission
def make_submission(predictions):
    submission = predictions.copy()
    submission['prediction'] = (submission['prediction'] > model_th) * 1
    return submission


# Get top k features by importance
def get_top_features(top_k=5):
    importances = model.get_feature_importance()
    names = model.feature_names_
    feature_dict = dict(zip(names, importances))
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = dict(sorted_features[:top_k])
    return top_features


# Plot predicted probas distribution
def plot_ditribution(predictions, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions['prediction'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    plt.title('Distribution of Model Prediction Scores', fontsize=16)
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlim(0, 1)
    plt.tight_layout()

    plt.savefig(save_path)
    logger.info(f'Distribution graph saved to {save_path}')
