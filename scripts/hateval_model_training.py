import os
import sys
import pandas as pd
import torch
import datasets
import transformers

sys.path.append('../modules/')

from custom_logger import get_logger
from model_utils import get_deberta_model
from training_metrics import compute_metrics_sklearn


def main():
    logger = get_logger('hateval_model_training')

    logger.info('Model fine-tuning on the hateval2019 data')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f'Using device: {device}')

    # Load data.
    # DATA_DIR = '/data/milanlp/moscato/personal_hate_bounds_data/hateval2019/'
    DATA_DIR = '/data1/moscato/personalised-hate-boundaries-data/data/hateval2019/'

    logger.info(f'Reading data from: {DATA_DIR}')

    training_data_df = pd.read_csv(os.path.join(DATA_DIR, f'hateval2019_en_train.csv'))[['text', 'HS']].rename(columns={'HS': 'labels'})
    test_data_df = pd.read_csv(os.path.join(DATA_DIR, f'hateval2019_en_test.csv'))[['text', 'HS']].rename(columns={'HS': 'labels'})

    train_ds = datasets.Dataset.from_dict(
        training_data_df
        .to_dict(orient='list')
    )
    test_ds = datasets.Dataset.from_dict(
        test_data_df
        .to_dict(orient='list')
    )

    # Load model.
    # MODEL_DIR = '/data/milanlp/huggingface/hub/'
    MODEL_DIR = '/data1/shared_models/'

    logger.info(f'Loading model from: {MODEL_DIR}')

    tokenizer, classifier = get_deberta_model(
        2,
        MODEL_DIR,
        device,
        use_custom_head=False,
        use_fast_tokenizer=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding='max_length',
            truncation=True,
            max_length=512,
            # return_tensors='pt'
        )
    
    # Format data (tokenization, formatting).
    tokenized_train_ds = train_ds.map(tokenize_function, batched=True).remove_columns('text')
    tokenized_test_ds = test_ds.map(tokenize_function, batched=True).remove_columns('text')

    tokenized_train_ds.set_format(
        type = "torch",
        columns = ["input_ids", "attention_mask", "labels"]
    )
    tokenized_test_ds.set_format(
        type = "torch",
        columns = ["input_ids", "attention_mask", "labels"]
    )

    # Train model.
    EXPERIMENT_ID = 'hateval_data_model_test_2'
    # MODEL_OUTPUT_DIR = f'/data/milanlp/moscato/personal_hate_bounds_data/trained_models/hateval2019_test/{EXPERIMENT_ID}/'
    MODEL_OUTPUT_DIR = f'/data1/moscato/personalised-hate-boundaries-data/models/hateval_data_model_test/{EXPERIMENT_ID}/'
    N_EPOCHS = 10

    logger.info(f'Model output directory: {MODEL_OUTPUT_DIR}')
    logger.info('Starting training')

    training_args = transformers.TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",  # Options: 'no', 'epoch', 'steps' (requires the `save_steps` argument to be set though).
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=1e-4,
        per_device_train_batch_size=8,  # Default: 8.
        gradient_accumulation_steps=1,  # Default: 1.
        per_device_eval_batch_size=8,  # Default: 8.
        num_train_epochs=N_EPOCHS,
        warmup_ratio=0.0,  # For linear warmup of learning rate.
        metric_for_best_model="f1",
        push_to_hub=False,
        logging_strategy='epoch',
        logging_first_step=True,
        logging_dir=f'../tensorboard_logs/{EXPERIMENT_ID}/',
        disable_tqdm=False
    )

    trainer = transformers.Trainer(
        model=classifier,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_test_ds,
        compute_metrics=compute_metrics_sklearn,
    )

    training_output = trainer.train()


if __name__ == '__main__':
    main()
