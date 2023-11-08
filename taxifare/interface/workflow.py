import os

import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from prefect import task, flow

from taxifare.interface.main import evaluate, preprocess, train
from taxifare.ml_logic.registry import mlflow_transition_model
from taxifare.params import *

@task
def preprocess_new_data(min_date: str, max_date: str):
    preprocessed_month=preprocess(min_date, max_date)
    return preprocessed_month

@task
def evaluate_production_model(min_date: str, max_date: str):
    old_mae=evaluate(min_date,max_date)
    return old_mae

@task
def re_train(min_date: str, max_date: str, split_ratio: str):
    new_mae=train(min_date,max_date,split_ratio)
    return new_mae

@task
def transition_model(current_stage: str, new_stage: str):
    ml_flow=mlflow_transition_model(current_stage,new_stage)
    return


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Build the Prefect workflow for the `taxifare` package. It should:
        - preprocess 1 month of new data, starting from EVALUATION_START_DATE
        - compute `old_mae` by evaluating the current production model in this new month period
        - compute `new_mae` by re-training, then evaluating the current production model on this new month period
        - if the new one is better than the old one, replace the current production model with the new one
        - if neither model is good enough, send a notification!
    """

    min_date = EVALUATION_START_DATE
    max_date = str(datetime.strptime(min_date, "%Y-%m-%d") + relativedelta(months=1)).split()[0]

    # Define the orchestration graph ("DAG")
    split_ratio=0.2
    preprocess_future = preprocess_new_data().submit(min_date,max_date)
    evaluate_future = evaluate_production_model().submit(min_date,max_date,wait_for=[preprocess_future]) # <-- task2 starts only after task1
    re_train_future = re_train().submit(min_date,max_date,split_ratio,wait_for=[preprocess_future,evaluate_future])
    # Compute your results as actual python object
    preprocess_result = preprocess_future.result()
    evaluate_result = evaluate_future.result()
    re_train_result = re_train_future.result()

    # Do something with the results (e.g. compare them)
    if re_train_result < evaluate_result:
        transition_model_future=transition_model().submit('Staging','Production',wait_for=[preprocess_future,evaluate_future,re_train_future])
    return



if __name__ == "__main__":
    # Actually launch your workflow
    train_flow()
