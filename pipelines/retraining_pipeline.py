# pipelines/retraining_pipeline.py
import schedule
import time
from src.train import train_model
from src.predict import predict_fixtures
from monitoring.drift_detection import detect_drift

def retraining_job():
    print("Starting retraining pipeline...")
    
    # Check for data drift
    drift_results = detect_drift()
    if any([result['drift_detected'] for result in drift_results.values()]):
        print("Data drift detected - retraining model")
        train_model()
        
        # Update predictions
        _, _, config = load_data()
        predict_fixtures(config['data_paths']['new_fixtures'])
        print("New predictions generated")
    else:
        print("No significant data drift - skipping retraining")
    
    print("Retraining pipeline completed")

if __name__ == "__main__":
    # Run weekly on Sunday at 2 AM
    schedule.every().sunday.at("02:00").do(retraining_job)
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour