import os
import sys
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import datasets
import transformers
from transformers import (AutoConfig, PretrainedConfig, AutoTokenizer,
    AutoModelForSequenceClassification, pipeline)
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

sys.path.append('../modules/')

from utils import DATETIME_FORMAT
from custom_logger import get_logger
from data_utils import generate_aggregated_labels_dataset
from model_utils import get_deberta_model
from training_metrics import compute_metrics_sklearn
from model_utils import freeze_model_weights, count_model_params
from training import WeightedLossTrainer

# MONICA
# DATASET_PATHS = {
#     'popquorn': '../data/samples/POPQUORN_offensiveness.csv',
#     'kumar': {
#         'train': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_train.csv',
#         'test': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_test.csv',
#     }
# }
# Bocconi HPC
DATASET_PATHS = {
    'popquorn': '../data/samples/POPQUORN_offensiveness.csv',
    'kumar': {
        # 'train': '/data/milanlp/moscato/personal_hate_bounds_data/kumar_processed_with_ID_and_full_perspective_clean_train.csv',
        # 'test': '/data/milanlp/moscato/personal_hate_bounds_data/kumar_processed_with_ID_and_full_perspective_clean_test.csv',
        'train': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_train.csv',
        'test': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_test.csv',
    }
}
SEPHEADS_SUBSAMPLED_TRAIN_DATASET_PATH = '/data1/moscato/personalised-hate-boundaries-data/models/sepheads_model_training_test_subsampling_2/training_data_subsampled.csv'
SEPHEADS_SUBSAMPLED_TEST_DATASET_PATH = '/data1/moscato/personalised-hate-boundaries-data/models/sepheads_model_training_test_subsampling_2/test_data_subsampled.csv'
DATASET_NAME = 'kumar'
SUBSAMPLE_MAJORITY_CLASS = False
MODEL_DIRS = {
    # 'microsoft/deberta-v3-base': '/data/milanlp/huggingface/hub/'
    'microsoft/deberta-v3-base': '/data1/shared_models/'
}
MODEL_ID = 'microsoft/deberta-v3-base'
FREEZE_ENCODER_PARAMS = False
EXPERIMENT_ID = 'majority_vote_model_sepheads_subsampled_data_test' # 'majority_vote_model_new_binarized_labels_1'  # f'test_model_{datetime.strftime(datetime.today(), DATETIME_FORMAT)}'
# MODEL_OUTPUT_DIR = f'/data/milanlp/moscato/personal_hate_bounds_data/trained_models/majority_vote_models/{EXPERIMENT_ID}/' 
MODEL_OUTPUT_DIR = f'/data1/moscato/personalised-hate-boundaries-data/models/{EXPERIMENT_ID}/'
CLASS_WEIGHTS = False
N_EPOCHS = 10

TESTING = False

# If set to `True`, the text IDs for training and testing are subset to those
# in the training and test data used to train the SepHeads model (as found in
# the indicated datasets).
USE_SEPHEADS_TRAINING_DATA = True


def main():
    logger = get_logger('train_majority_vote_model')

    logger.info(
        f'Training majority vote model on dataset: {DATASET_NAME}'
        f' | Experiment ID: {EXPERIMENT_ID}'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f'Training on device: {device}')

    # Read data.
    training_data, test_data = generate_aggregated_labels_dataset(
        DATASET_NAME,
        DATASET_PATHS[DATASET_NAME]['train'],
        DATASET_PATHS[DATASET_NAME]['test'],
        subsample_majority_class=SUBSAMPLE_MAJORITY_CLASS
    )

    if USE_SEPHEADS_TRAINING_DATA:
        logger.info(
            'Restricting the data to that used to train the SepHeads model'
        )

        # Get the text IDs used to train the SepHeads model reading them from
        # the corresponding datasets.
        sepheads_training_text_ids = pd.read_csv(
            SEPHEADS_SUBSAMPLED_TRAIN_DATASET_PATH
        )['text_id'].drop_duplicates().tolist()
        
        sepheads_test_text_ids = pd.read_csv(
            SEPHEADS_SUBSAMPLED_TEST_DATASET_PATH
        )['text_id'].drop_duplicates().tolist()

        # Subset the training and test datasets to the selected text IDs.
        training_data = training_data[
            training_data['text_id'].isin(sepheads_training_text_ids)
        ].drop(columns=['text_id']).reset_index(drop=True)
        test_data = test_data[
            test_data['text_id'].isin(sepheads_test_text_ids)
        ].drop(columns=['text_id']).reset_index(drop=True)

    # Put datasets into Hugging Face datasets.
    if TESTING:
        logger.warning('Working in TESTING mode')

        # Select only 1000 training and 100 test samples if testing.
        train_ds = datasets.Dataset.from_dict(
            training_data
            .iloc[:10000]  # Testing!
            .to_dict(orient='list')
        )
        test_ds = datasets.Dataset.from_dict(
            test_data
            .iloc[:1000]  # Testing!
            .to_dict(orient='list')
        )
    else:
        train_ds = datasets.Dataset.from_dict(
            training_data
            .to_dict(orient='list')
        )
        test_ds = datasets.Dataset.from_dict(
            test_data
            .to_dict(orient='list')
        )

    # Instantiate model.
    num_labels = training_data['label'].unique().shape[0]

    tokenizer, classifier = get_deberta_model(
        num_labels,
        MODEL_DIRS[MODEL_ID],
        device,
        use_custom_head=False,
        pooler_out_features=768,
        pooler_drop_prob=0.,
        classifier_drop_prob=0.1,
        use_fast_tokenizer=False
    )

    # Tokenize datasets.
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=512,  # Beyond the 99th percentile of length for Kumar data.
            # return_tensors='pt'
        )

    logger.info(f'Tokenizing datasets')

    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

    logger.info(
        f'Training dataset size: {len(train_ds)}'
        f' | Test dataset size: {len(test_ds)}'
    )

    # Instantiate data collator (to pass to the trainer).
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # Choose whether freezing the encoder's parameters or not.
    if FREEZE_ENCODER_PARAMS:
        freeze_model_weights(
            classifier, trainable_modules=['classifier']
        )

    n_params_total, n_params_trainable = count_model_params(classifier)

    logger.info(
        f'N params: {n_params_total} | N trainable params: {n_params_trainable}'
    )

    # Select training mode for the model.
    classifier.train()

    logger.info(
        f'Training mode selected: {classifier.training}'
    )

    # Instantitate `Trainer` object and train model.
    logger.info('Training model')

    if not os.path.exists(MODEL_OUTPUT_DIR):
        logger.info(f'Creating model output dir: {MODEL_OUTPUT_DIR}')

        os.makedirs(MODEL_OUTPUT_DIR)

    logger.info(
        f'N epochs: {N_EPOCHS} | Model output dir: {MODEL_OUTPUT_DIR}'
    )

    training_args = transformers.TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",  # Options: 'no', 'epoch', 'steps' (requires the `save_steps` argument to be set though).
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=1e-6,  # Default: 1e-4.
        per_device_train_batch_size=64,  # Default: 8.
        gradient_accumulation_steps=1,  # Default: 1.
        per_device_eval_batch_size=32,  # Default: 8.
        num_train_epochs=N_EPOCHS,
        warmup_ratio=0.0,  # For linear warmup of learning rate.
        metric_for_best_model="f1",
        push_to_hub=False,
        # label_names=list(roberta_classifier.config.id2label.keys()),
        logging_strategy='epoch',
        logging_first_step=True,
        logging_dir=f'../tensorboard_logs/{EXPERIMENT_ID}/',
        # logging_steps=10,
        disable_tqdm=True
    )

    if CLASS_WEIGHTS:
        logger.info('Training with custom class weights')

        class_weights_from_frequencies = class_weights_from_frequencies = (
            training_data.groupby('label')['instance_id'].count().sort_index(ascending=True)
            / len(training_data)
        ).to_list()
    
        trainer = WeightedLossTrainer(
            class_weights=torch.tensor(class_weights_from_frequencies).to(device=device),
            model=classifier,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_test_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_sklearn,
        )
    else:
        logger.info('Training without class weights')
        
        trainer = transformers.Trainer(
            model=classifier,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_test_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_sklearn,
        )

    training_output = trainer.train()

    logger.info('Training over')


if __name__ == '__main__':
    main()
