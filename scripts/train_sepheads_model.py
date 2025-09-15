import sys
from copy import deepcopy
import pandas as pd
import torch
import datasets
import transformers

sys.path.append('../modules/')

from custom_logger import get_logger
from data_utils import subsample_dataset
from model_utils import get_deberta_model
from models import DebertaWithAnnotatorHeads
from training_metrics import compute_metrics_sklearn


DATASET_PATHS = {
    'popquorn': '../data/samples/POPQUORN_offensiveness.csv',
    'kumar': {
        'train': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_train.csv',
        'test': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/kumar_processed_with_ID_and_full_perspective_clean_test.csv',
        'annotators_data': '/data1/moscato/personalised-hate-boundaries-data/data/kumar_perspective_clean/annotators_data.csv',
        # 'train': '/data/milanlp/moscato/personal_hate_bounds_data/kumar_processed_with_ID_and_full_perspective_clean_train.csv',
        # 'test': '/data/milanlp/moscato/personal_hate_bounds_data/kumar_processed_with_ID_and_full_perspective_clean_test.csv',
        # 'annotators_data': '/data/milanlp/moscato/personal_hate_bounds_data/annotators_data.csv'
    },
    # Data used for the original training./
    # 'mhs': {
    #     'train': '/data1/moscato/personalised-hate-boundaries-data/data/measuring_hate_speech_data_clean/mhs_clean_train.csv',
    #     'test': '/data1/moscato/personalised-hate-boundaries-data/data/measuring_hate_speech_data_clean/mhs_clean_test.csv'
    # }
    # New data (more samples, 10 samples per annotator.
    'mhs': {
        'train': '/data1/moscato/personalised-hate-boundaries-data/data/measuring_hate_speech_data_clean/mhs_clean_train_10_samples_per_annotator.csv',
        'test': '/data1/moscato/personalised-hate-boundaries-data/data/measuring_hate_speech_data_clean/mhs_clean_test_10_samples_per_annotator.csv'
    }
}
DATASET_NAME = 'mhs'
MODEL_DIRS = {
    # 'microsoft/deberta-v3-base': '/data/milanlp/huggingface/hub/'
    'microsoft/deberta-v3-base': '/data1/shared_models/'
}
MODEL_ID = 'microsoft/deberta-v3-base'
EXPERIMENT_ID = 'sepheads_model_training_mhs_enlarged_dataset_4'
# MODEL_OUTPUT_DIR = f'/data/milanlp/moscato/personal_hate_bounds_data/trained_models/majority_vote_models/{EXPERIMENT_ID}/'
MODEL_OUTPUT_DIR = f'/data1/moscato/personalised-hate-boundaries-data/models/{EXPERIMENT_ID}/'
N_EPOCHS = 10

N_ANNOTATORS_TEST = None
OPTIMAL_N_TRAINING_DATAPOINTS = None


def main():
    logger = get_logger('sepheads_model_training')

    logger.info(
        f'Training SepHeads model on dataset: {DATASET_NAME}'
        f' | Experiment ID: {EXPERIMENT_ID}'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data.
    logger.info(f'Loading data from: {DATASET_PATHS[DATASET_NAME]}')

    training_data = pd.read_csv(DATASET_PATHS[DATASET_NAME]['train'])
    test_data = pd.read_csv(DATASET_PATHS[DATASET_NAME]['test'])

    # Restrict to the first `N_ANNOTATORS_TEST` annotators for testing.
    if N_ANNOTATORS_TEST is not None:
        logger.warning(f'Testing with {N_ANNOTATORS_TEST} annotators')

        annotator_ids = sorted(training_data['annotator_id'].unique())[:N_ANNOTATORS_TEST]

        training_data = training_data[training_data['annotator_id'].isin(annotator_ids)].reset_index(drop=True)
        test_data = test_data[test_data['annotator_id'].isin(annotator_ids)].reset_index(drop=True)
    else:
        if OPTIMAL_N_TRAINING_DATAPOINTS is not None:
            logger.info(
                f'Subsampling data to {OPTIMAL_N_TRAINING_DATAPOINTS}'
                ' training datapoints'
            )

            training_data, test_data = subsample_dataset(
                training_data,
                test_data,
                OPTIMAL_N_TRAINING_DATAPOINTS,
                DATASET_PATHS[DATASET_NAME]['annotators_data']
            )

        annotator_ids = sorted(training_data['annotator_id'].unique())

    logger.info(
        f'N annotators: {len(annotator_ids)} | N training samples: {len(training_data)}'
        f' | N test samples: {len(test_data)}'
    )

    # Instantiate the DeBERTa text encoder.
    logger.info('Instantiating the SepHeads model')

    num_labels = training_data['toxic_score'].unique().shape[0]

    logger.info(f'N labels found in training data: {num_labels}')

    deberta_tokenizer, deberta_model = get_deberta_model(
        num_labels,
        MODEL_DIRS[MODEL_ID],
        device,
        use_custom_head=False,
        pooler_out_features=768,  # Default: 768.
        pooler_drop_prob=0.0,  # Default: 0.0
        classifier_drop_prob=0.1,  # Default: 0.1
        use_fast_tokenizer=False
    )

    deberta_with_annotator_heads_model = DebertaWithAnnotatorHeads(
        deberta_encoder=deepcopy(deberta_model.deberta),
        deberta_pooler=deepcopy(deberta_model.pooler),
        deberta_dropout=deepcopy(deberta_model.dropout),
        num_labels=num_labels,
        annotator_ids=annotator_ids,
    )

    del deberta_model

    # Create tokenized datasets.
    logger.info('Creating tokenized datasets')

    def tokenize_function(examples):
        return deberta_tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=512,
            # return_tensors='pt'
        )

    tokenized_training_data = (
        # Create datast object from the DataFrame.
        datasets.Dataset.from_dict(
            training_data[[
                'text',
                'toxic_score',
                'annotator_id'
            ]].rename(
                columns={
                    'toxic_score': 'label',
                    'annotator_id': 'annotator_ids',
                }
            )
            .to_dict(orient='list')
        )
        # Tokenize.
        .map(tokenize_function, batched=True)
        # Remove useless column.
        .remove_columns("text")
        .shuffle()
        .flatten_indices()
    )

    tokenized_test_data = (
        # Create datast object from the DataFrame.
        datasets.Dataset.from_dict(
            test_data[[
                'text',
                'toxic_score',
                'annotator_id'
            ]].rename(
                columns={
                    'toxic_score': 'label',
                    'annotator_id': 'annotator_ids',
                }
            )
            .to_dict(orient='list')
        )
        # Tokenize.
        .map(tokenize_function, batched=True)
        # Remove useless column.
        .remove_columns("text")
    )

    # Instantiate training.
    logger.info('Training starting')

    training_args = transformers.TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",  # Options: 'no', 'epoch', 'steps' (requires the `save_steps` argument to be set though).
        save_total_limit=2,
        load_best_model_at_end=True,
        learning_rate=1e-6,
        per_device_train_batch_size=16,  # Default: 8.
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

    data_collator = transformers.DataCollatorWithPadding(tokenizer=deberta_tokenizer)

    trainer = transformers.Trainer(
        model=deberta_with_annotator_heads_model,
        args=training_args,
        train_dataset=tokenized_training_data,
        eval_dataset=tokenized_test_data,
        data_collator=data_collator,
        tokenizer=deberta_tokenizer,
        compute_metrics=compute_metrics_sklearn,
    )

    training_output = trainer.train()

    logger.info('Training finished')


if __name__ == '__main__':
    main()
