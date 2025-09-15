import os
from copy import deepcopy
import numpy as np
import safetensors
from custom_logger import get_logger
from models import DebertaWithAnnotatorHeads


def count_model_params(model):
    """
    """
    n_params_total = sum([p.numel() for p in model.parameters()])
    n_params_trainable = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )

    return n_params_total, n_params_trainable


def freeze_model_weights(model, trainable_modules=['classifier']):
    """
    Freezes the weights in all the submodules of `model` whose name
    don't appear in the `trainable_modules` list. The submodules for which the
    weigths are frozen are also set to inference mode (dropout layers are
    deactivated so the submodule's output is deterministic).
    """
    logger = get_logger('freeze_encoder_weights')

    for submodule in model.named_children():
        if submodule[0] not in trainable_modules:
            for p in submodule[1].parameters():
                p.requires_grad = False

            # Set all the PyTorch modules that are not meant to remain
            # trainable to inference mode.
            submodule[1].train(mode=False)

        logger.info(
            f'Module: {submodule[0]}'
            f' | N parameters: {sum([p.numel() for p in submodule[1].parameters()])}'
            f' | Parameters trainable: {all([p.requires_grad for p in submodule[1].parameters()])}'
            f' | Training mode: {submodule[1].training}'
        )


def switch_training_mode_submodules(mode, model, submodules):
    """
    Given a model and a list of its submodules (not necessarily all of them),
    applies the `mode` training mode to that submodules. Possible modes:
      * `mode='train'`: the submodules are put in training mode (dropout and
                        batch normalization on).
      * `mode='eval'`: the submodules are put in evaluation mode (dropout and
                       batch normalization off).
    """
    if mode not in ['train', 'eval']:
        raise Exception(
            f"Mode {mode} not implemented (possible values: 'train', 'eval')"
        )

    # Convert the mode into a bool indicating whether we want training mode or
    # not.
    train_bool = True if mode == 'train' else False

    all_submodules = [s[0] for s in model.named_children()]

    for submodule in submodules:
        if submodule not in all_submodules:
            raise Exception(
                f"Submodule {submodule} not found among the model's"
                f" submodules ({all_submodules})"
            )

        # Apply training mode to submodule.
        getattr(model, submodule).train(mode=train_bool)


import torch
from transformers import (AutoConfig, PretrainedConfig, AutoTokenizer,
    AutoModelForSequenceClassification, DebertaForSequenceClassification,
    DebertaV2ForSequenceClassification)
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout


def get_deberta_model(
        num_labels,
        model_dir,
        device,
        use_custom_head=True,
        pooler_out_features=None,  # Default: 768.
        pooler_drop_prob=None,  # Default: 0.0
        classifier_drop_prob=None,  # Default: 0.1
        use_fast_tokenizer=False
    ):
    """
    Returns an instance of DeBERTa loading an encoder with pre-trained weights
    from the Hugging Face hub (model ID: `microsoftmicrosoft/deberta-v3-base`).
    """
    logger = get_logger('get_deberta_model')

    model_id = 'microsoft/deberta-v3-base'

    # Instantiate tokenizer.
    logger.info('Instantiating DeBERTa tokenizer')

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=model_dir,
        use_fast=use_fast_tokenizer
    )

    if use_custom_head:
        logger.info(
            'Instantiating DeBERTa model with custom classification head'
        )

        # Config for the encoder.
        deberta_classifier_config = AutoConfig.from_pretrained(
            model_id,
            finetuning_task="text-classification",
            id2label={
                i: label
                for i, label in enumerate(range(num_labels))
            },
            label2id={
                label: i
                for i, label in enumerate(range(num_labels))
            }
        )

        # Config for the "classification head" for the DeBERTa model (it's not
        # an object in itself - layers must be instantiated manually one by
        # one).
        if any([
            s is None
            for s in [pooler_out_features, pooler_drop_prob, classifier_drop_prob]
        ]):
            raise Exception(
                'The `pooler_out_features`, `pooler_drop_prob` and'
                ' `classifier_drop_prob` options cannot be `None` if a custom'
                ' classification head is to be used'
            )

        deberta_classification_head_config = PretrainedConfig()

        deberta_classification_head_config.pooler_in_features = deberta_classifier_config.hidden_size
        deberta_classification_head_config.pooler_out_features = pooler_out_features
        deberta_classification_head_config.pooler_drop_prob = pooler_drop_prob
        deberta_classification_head_config.classifier_in_features = deberta_classification_head_config.pooler_out_features
        deberta_classification_head_config.classifier_out_features = num_labels
        deberta_classification_head_config.classifier_drop_prob = classifier_drop_prob

        # Instantiate DeBERTa model.
        classifier = DebertaV2ForSequenceClassification.from_pretrained(
            model_id,
            config=deberta_classifier_config,
            cache_dir=model_dir,
        )

        # Substitute the default "classification head" with a custom one.
        classifier.pooler.dense = torch.nn.Linear(
            in_features=deberta_classification_head_config.pooler_in_features,
            out_features=deberta_classification_head_config.pooler_out_features,
            bias=True
        )
        classifier.pooler.dropout = StableDropout(
            drop_prob=deberta_classification_head_config.pooler_drop_prob
        )
        classifier.classifier = torch.nn.Linear(
            in_features=deberta_classification_head_config.classifier_in_features,
            out_features=deberta_classification_head_config.classifier_out_features,
            bias=True
        )
        classifier.dropout = StableDropout(
            drop_prob=deberta_classification_head_config.classifier_drop_prob
        )

    else:
        logger.info(
            'Instantiating DeBERTa model with default classification head'
        )

        classifier = DebertaV2ForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            cache_dir=model_dir
        )

    # Send classifier to the specified device.
    classifier.to(device=device)

    return tokenizer, classifier


def load_sepheads_model_safetensors(
        checkpoint_path,
        annotator_ids,
        deberta_model_dir,
        device
    ):
    """
    Load the checkpoint of a fine-tuned SepHeads model using the `safetensors`
    library (i.e. NOT with a Hugging Face-compliant model object). Returns
    a `models.DebertaWithAnnotatorHeads` object.

    Notes:
      * We ASSUME we only have 2 labels and that we're using the default
        DeBERTa v2 architecture for the classification heads.
      * The annotator IDs must be known before loading the model.
    """
    _, deberta_model = get_deberta_model(
        2,
        deberta_model_dir,
        device,
        use_custom_head=False,
        use_fast_tokenizer=False
    )

    model = DebertaWithAnnotatorHeads(
        deberta_encoder=deepcopy(deberta_model.deberta),
        deberta_pooler=deepcopy(deberta_model.pooler),
        deberta_dropout=deepcopy(deberta_model.dropout),
        num_labels=2,
        annotator_ids=annotator_ids,
    )

    del deberta_model

    # Pedantic way.
    # with safetensors.safe_open(
    #     os.path.join(MODEL_OUTPUT_DIR, f'checkpoint-{checkpoint}', 'model.safetensors'),
    #     framework="pt",
    #     device=device.type
    # ) as f:
    #     state_dict = {}
        
    #     for key in f.keys():
    #         state_dict[key] = f.get_tensor(key)
    
    # Quick way. The outputs are missing and unexpected module names (should
    # both be empty).
    missing, unexpected = safetensors.torch.load_model(
        model=model,
        filename=os.path.join(checkpoint_path, 'model.safetensors')
    )
    
    if not len(missing) == 0:
        raise Exception('Missing modules found when loading the saved model')
    
    if not len(unexpected) == 0:
        raise Exception('Unexpected modules found when loading the saved model')

    return model


def compute_combined_prediction(
        sepheads_preds,
        boundary_model_preds,
        confidence_scores,
        confidence_threshold
    ):
    """ 
    Parameters
    ----------
    sepheads_preds :
        
    boundary_model_preds :
    
    confidence_threshold : float
        The confidence score theshold above which the prediction from
        GPT overrides the one from SepHeads.
    """
    # return np.where(
    #     confidence_scores <= confidence_threshold,
    #     sepheads_preds,
    #     boundary_model_preds,
    # )
    return np.where(
        (confidence_scores >= confidence_threshold) & (boundary_model_preds == 1),
        boundary_model_preds,
        sepheads_preds
    )
