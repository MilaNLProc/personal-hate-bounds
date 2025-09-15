import os
from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import trange
import transformers
from torch.utils.tensorboard import SummaryWriter
from custom_logger import get_logger
from training_metrics import compute_metrics_sklearn_training
from pytorch_utils import send_batch_to_device


class WeightedLossTrainer(transformers.Trainer):
    """
    Source: https://discuss.huggingface.co/t/how-can-i-use-class-weights-when-training/1067/6

    Additions: the `num_items_in_batch` keyword argument has been added. The
    logic is:
        * If `num_items_in_batch` is `None`, then the loss function is a
          cross entropy loss computing the mean over the samples in the batch
          (default).
        * Else, the loss function is computed as a SUM over the samples, and
          the result is divided by `num_items_in_batch` (this should be the
          intended use in the Transformers `Trainer` object, although how this
          works is not explicit in the code thereof).

    Code for the original `Trainer` class for comparison:
    https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = get_logger('weighted_loss_trainer')

        self.class_weights = class_weights

        self.logger.info(
            f'Initialized WeightedLossTrainer with class weights:'
            f' {self.class_weights}'
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # compute custom loss
        if num_items_in_batch is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights, reduction='sum'
            )

            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            ) / num_items_in_batch
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)

            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

def generate_log_message_metrics(epoch_counter, training_history):
    """
    """
    return f'Epoch: {epoch_counter}' + ''.join([
        f' | {metric}: {values[-1]}'
        for metric, values in training_history.items()
    ])


def training_step(
        training_data,
        model,
        loss_fn,
        optimizer,
    ):
    """
    """
    # Unpack the training data.
    x_train, y_train = training_data

    # Compute the loss function on the training data.
    y_pred = model(**x_train).logits
    
    training_loss = loss_fn(y_pred, y_train)

    # Reset gradients and recompute it.
    optimizer.zero_grad()

    training_loss.backward()

    # Perform an optimization step.
    optimizer.step()

    return training_loss.detach()
    

def train_model(
        model,
        tokenized_training_ds,
        tokenized_test_ds,
        optimizer,
        n_epochs,
        batch_size,
        device,
        loss_fn=torch.nn.CrossEntropyLoss(),
        training_history=None,
        checkpointing_period_epochs=None,
        model_dir=None,
        checkpoint_id=None,
        save_final_model=False,
        tensorboard_log_dir=None
    ):
    """
    """
    logger = get_logger('train_model')

    logger.info('Training model')

    # Preliminary steps for model checkpointing.
    if (checkpointing_period_epochs is not None) or save_final_model:
        # Create directory in which to save the model, if it doesn't exist.
        if not os.path.exists(model_dir):
            logger.info(f'Creating directory in which to save model: {model_dir}')

            os.makedirs(model_dir)
        
        # Save model's (hyper)parameters.
        model_params_path = os.path.join(model_dir, 'model_params.json')

        with open(model_params_path, 'w') as f:
            json.dump(model.get_params_dict(), f)

    # Counter for the number of GRADIENT DESCENT STEPS performed in the
    # current training run (correspnding to the total number of batches the
    # model is trained upon across all epochs) - useful if learning rate
    # scheduling is used.
    update_counter = 0

    # Initialize a new training history if none is passed as input (training
    # from scratch).
    if training_history is None:
        epoch_counter = 0

        training_history = defaultdict(list)

    else:
        n_history_entries = len(
            training_history[list(training_history.keys())[0]]
        )

        # If an empty training history is passed as input (it could be
        # convenient code-wise), start from the first epoch.
        if n_history_entries == 0:
            epoch_counter = 0

            training_history = defaultdict(list)

        # If a non-empty training history is passed as input, resume training
        # from the last epoch.
        else:
            # Resume training from the last epoch, as inferred by the length
            # of the provided training history.
            # Note: by convention, the training history contains data for the
            #       past epochs PLUS THE INITIAL METRICS.
            epoch_counter = n_history_entries - 1  

            logger.info(f'Resuming training from epoch {epoch_counter}')

            if 'test_loss' in training_history.keys():
                if tokenized_test_ds is None:
                    raise Exception(
                        'Validation data was used in previous training, '
                        'please keep using it'
                    )
            else:
                if tokenized_test_ds is not None:
                    raise Exception(
                        'No validation data was used in previous training, '
                        'please keep not using it'
                    )
                
    # Instantiate Tensorboard writer if needed.
    if tensorboard_log_dir is not None:
        writer = SummaryWriter(
            log_dir=tensorboard_log_dir
        )
    else:
        writer = None

    # Put the tokenized training and test data into a data loader for batched
    # training/inference.
    training_loader = torch.utils.data.DataLoader(
        tokenized_training_ds.select_columns(
            ['label', 'input_ids', 'token_type_ids', 'attention_mask']
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    if tokenized_test_ds is not None:
        test_loader = torch.utils.data.DataLoader(
            tokenized_test_ds.select_columns(
                ['label', 'input_ids', 'token_type_ids', 'attention_mask']
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

    # If training the model from scratch, add all the metrics before training
    # starts to the training history.
    if epoch_counter == 0:
        print('Computing initial metrics')

        # For consistency with the training phase, let's compute the training
        # loss and accuracy over batches and then average it.
        training_loss_batches = []
        training_metrics_batches = defaultdict(list)

        n_batches_test = 10

        for i, batch in enumerate(training_loader):
            send_batch_to_device(batch, device)

            training_targets = batch['label']

            del batch['label']

            training_batch = batch

            with torch.no_grad():
                pred = model(**training_batch).logits

            training_loss_batch = loss_fn(
                pred, training_targets
            )

            # Extract the predicted labels and sent them to cpu along with the
            # targets (needed to compute the evaluation metrics).
            predicted_labels = torch.argmax(pred, dim=-1).cpu()
            training_targets = training_targets.cpu()

            training_metrics_batch = compute_metrics_sklearn_training(
                predicted_labels, training_targets
            )

            training_loss_batches.append(training_loss_batch)
            
            for metric, value in training_metrics_batch.items():
                training_metrics_batches[metric].append(value)

            if i == n_batches_test:
                break

        training_loss = torch.tensor(training_loss_batches).mean()
        training_metrics = {
            metric: np.mean(values)
            for metric, values in training_metrics_batches.items()
        }

        # Update training history.
        training_history['training_loss'].append(training_loss)

        for metric, value in training_metrics.items():
            training_history[f'training_{metric}'].append(value)

        training_history['learning_rate'].append(
            optimizer.state_dict()['param_groups'][0]['lr']
        )

        # Compute test loss and metrics.
        if tokenized_test_ds is not None:
            test_loss_batches = []
            test_metrics_batches = defaultdict(list)

            for i, batch in enumerate(test_loader):
                send_batch_to_device(batch, device)

                test_targets = batch['label']

                del batch['label']

                test_batch = batch

                with torch.no_grad():
                    pred = model(**test_batch).logits

                test_loss_batch = loss_fn(
                    pred, test_targets
                )

                # Extract the predicted labels and sent them to cpu along with
                # the targets (needed to compute the evaluation metrics).
                predicted_labels = torch.argmax(pred, dim=-1).cpu()
                test_targets = test_targets.cpu()

                test_metrics_batch = compute_metrics_sklearn_training(
                    predicted_labels, test_targets
                )

                test_loss_batches.append(test_loss_batch)
                
                for metric, value in test_metrics_batch.items():
                    test_metrics_batches[metric].append(value)

                if i == n_batches_test:
                    break

            test_loss = torch.tensor(test_loss_batches).mean()
            test_metrics = {
                metric: np.mean(values)
                for metric, values in test_metrics_batches.items()
            }

            # Update training history.
            training_history['test_loss'].append(test_loss)

            for metric, value in test_metrics.items():
                training_history[f'test_{metric}'].append(value)
        else:
            test_loss = None
            test_metrics = None

        logger.info(
            generate_log_message_metrics(epoch_counter, training_history)
        )

        # Write initial metrics to Tensorboard logs.
        if writer is not None:
            for metric, values in training_history.items():
                if metric == 'learning_rate':
                    writer.add_scalar(
                        'LR/train',
                        values[-1],
                        epoch_counter
                    )
                elif 'train' in metric:
                    metric_name = metric.replace('train_', '')

                    writer.add_scalar(
                        f'{metric_name}/train',
                        values[-1],
                        epoch_counter
                    )
                else:
                    metric_name = metric.replace('test_', '')

                    writer.add_scalar(
                        f'{metric_name}/test',
                        values[-1],
                        epoch_counter
                    )

    # Reset the data loaders.
    training_loader = torch.utils.data.DataLoader(
        tokenized_training_ds.select_columns(
            ['label', 'input_ids', 'token_type_ids', 'attention_mask']
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    if tokenized_test_ds is not None:
        test_loader = torch.utils.data.DataLoader(
            tokenized_test_ds.select_columns(
                ['label', 'input_ids', 'token_type_ids', 'attention_mask']
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

    # Training loop.
    with trange(n_epochs) as pbar:
        for i in pbar:
            epoch_counter += 1

            training_loss_batches = []
            training_metrics_batches = defaultdict(list)

            for batch in training_loader:
                send_batch_to_device(batch, device)

                training_targets = batch['label']

                del batch['label']

                training_batch = batch

                training_loss_batch = training_step(
                    (training_batch, training_targets),
                    model,
                    loss_fn,
                    optimizer,
                )

                # Extract the predicted labels and sent them to cpu along with
                # the targets (needed to compute the evaluation metrics).
                predicted_labels = torch.argmax(pred, dim=-1).cpu()
                training_targets = training_targets.cpu()

                training_metrics_batch = compute_metrics_sklearn_training(
                    predicted_labels, training_targets
                )

                training_loss_batches.append(training_loss_batch)
                
                for metric, value in training_metrics_batch.items():
                    training_metrics_batches[metric].append(value)

            # Training loss and accuracy for one epoch is computed as the
            # average training loss over the batches.
            training_loss = torch.tensor(training_loss_batches).mean()
            training_metrics = {
                metric: np.mean(values)
                for metric, values in training_metrics_batches.items()
            }

            # Update training history.
            training_history['training_loss'].append(training_loss)

            for metric, value in training_metrics.items():
                training_history[f'training_{metric}'].append(value)

            training_history['learning_rate'].append(
                optimizer.state_dict()['param_groups'][0]['lr']
            )

            if tokenized_test_ds is not None:
                test_loss_batches = []
                test_metrics_batches = defaultdict(list)

                for i, batch in enumerate(test_loader):
                    send_batch_to_device(batch, device)

                    test_targets = batch['label']

                    del batch['label']

                    test_batch = batch

                    with torch.no_grad():
                        pred = model(**test_batch).logits

                    test_loss_batch = loss_fn(
                        pred, test_targets
                    )

                    # Extract the predicted labels and sent them to cpu along
                    # with the targets (needed to compute the evaluation
                    # metrics).
                    predicted_labels = torch.argmax(pred, dim=-1).cpu()
                    test_targets = test_targets.cpu()

                    test_metrics_batch = compute_metrics_sklearn_training(
                        predicted_labels, test_targets
                    )

                    test_loss_batches.append(test_loss_batch)
                    
                    for metric, value in test_metrics_batch.items():
                        test_metrics_batches[metric].append(value)

                    if i == n_batches_test:
                        break

                test_loss = torch.tensor(test_loss_batches).mean()
                test_metrics = {
                    metric: np.mean(values)
                    for metric, values in test_metrics_batches.items()
                }

                # Update training history.
                training_history['test_loss'].append(test_loss)

                for metric, value in test_metrics.items():
                    training_history[f'test_{metric}'].append(value)
            else:
                test_loss = None
                test_metrics = None
                

            pbar.set_postfix(**{
                metric: values[-1]
                for metric, values in training_history.items()
            })

            # # Write metrics to Tensorboard logs.
            # if writer is not None:
            #     writer.add_scalar(
            #         'Loss/train',
            #         training_history['training_loss'][-1],
            #         epoch_counter
            #     )
            #     writer.add_scalar(
            #         'Accuracy/train',
            #         training_history['training_accuracy'][-1],
            #         epoch_counter
            #     )
            #     writer.add_scalar(
            #         'LR/train',
            #         training_history['learning_rate'][-1],
            #         epoch_counter
            #     )
            #     writer.add_scalar(
            #         'Loss/val',
            #         training_history['val_loss'][-1],
            #         epoch_counter
            #     )
            #     writer.add_scalar(
            #         'Accuracy/val',
            #         training_history['val_accuracy'][-1],
            #         epoch_counter
            #     )

            # Model checkpointing (if required).
            if (
                (checkpointing_period_epochs is not None)
                and (epoch_counter % checkpointing_period_epochs == 0)
            ):
                # Save model/optimizer checkpoint.
                checkpoint_path = os.path.join(
                    model_dir,
                    checkpoint_id + f'_epoch_{epoch_counter}.pt'
                )

                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_history': training_history
                    },
                    checkpoint_path
                )

    return training_history