from datetime import datetime
import json
import logging
import os
import pandas as pd
import sys
import time
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

sys.path.append(os.path.abspath('./src'))
from preprocessing import load_train_data, run_preproc
from scorer import make_pred, get_top_features, plot_ditribution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProcessingService:
    def __init__(self):
        logger.info('Initializing ProcessingService...')
        self.input_dir = '/app/input'
        self.output_dir = '/app/output'
        self.train = load_train_data()
        logger.info('Service initialized')

    def process_single_file(self, file_path):
        try:
            logger.info('Processing file: %s', file_path)
            input_df = pd.read_csv(file_path).drop(columns=['name_1', 'name_2', 'street', 'post_code'])

            logger.info('Starting preprocessing')
            processed_df = run_preproc(self.train, input_df)
            
            logger.info('Making prediction')
            submission = make_pred(processed_df, file_path)


            top_k = 5
            logger.info(f'Getting top {top_k} features')
            features = get_top_features(top_k=top_k)
            top_features_file_name = f"top_{top_k}_features.json"
            with open(os.path.join(self.output_dir, top_features_file_name), "w+") as f:
                json.dump(features, f)
            
            logger.info('Prepraring submission file')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_file_name = os.path.basename(file_path)
            output_filename = f"predictions_{timestamp}_{data_file_name}"
            submission.to_csv(os.path.join(self.output_dir, output_filename), index=False)

            logger.info('Plotting predictions distribution')
            graph_filename = f"distribution_{timestamp}_{os.path.splitext(data_file_name)[0]}.png"
            plot_ditribution(submission, os.path.join(self.output_dir, graph_filename))

            logger.info('Predictions saved to: %s', output_filename)

        except Exception as e:
            logger.error('Error processing file %s: %s', file_path, e, exc_info=True)
            return


class FileHandler(FileSystemEventHandler):
    def __init__(self, service):
        self.service = service

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".csv"):
            logger.debug('New file detected: %s', event.src_path)
            self.service.process_single_file(event.src_path)

if __name__ == "__main__":
    logger.info('Starting ML scoring service...')
    service = ProcessingService()
    observer = PollingObserver()
    observer.schedule(FileHandler(service), path=service.input_dir, recursive=False)
    observer.start()
    logger.info('File observer started')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
        observer.stop()
    observer.join()
