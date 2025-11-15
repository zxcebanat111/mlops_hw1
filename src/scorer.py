from catboost import CatBoostClassifier
import logging
import matplotlib.pyplot as plt
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

# Make prediction
def make_pred(dt, path_to_file):

    # Make submission dataframe
    submission = pd.DataFrame({
        'index':  pd.read_csv(path_to_file).index,
        'prediction': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Return proba for positive class
    return submission


# Get top k features by importance
def get_top_features(top_k=5):
    importances = model.get_feature_importance()
    names = model.feature_names_
    feature_dict = dict(zip(names, importances))
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = dict(sorted_features[:top_k])
    return top_features


# Plot predicted labels distribution
def plot_ditribution(predictions, save_path):
    stats = predictions.groupby('prediction')['index'].count()

    plt.figure(figsize=(8, 6))
    plt.bar(x=stats.index.values, height=stats.values, color=['skyblue', 'lightcoral'])

    plt.xlabel('Target Value')
    plt.ylabel('Count')
    plt.title('Distribution of Target Variable')
    plt.xticks(stats.index.values, ['Not Fraud (0)', 'Fraud (1)'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path)
    logger.info(f'Graph saved to {save_path}')
