import torch
import transformers


class DebertaWithAnnotatorHeads(torch.nn.Module):
    """
    A `DebertaV2ForSequenceClassification` model, with
    annotator-specific classification heads.
    """
    def __init__(
        self,
        deberta_encoder,
        deberta_pooler,
        deberta_dropout,
        num_labels,
        annotator_ids,
        **kwargs
    ):
        """
        Parameters
        ----------
        deberta_encoder : transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model
            DeBERTa model encoder, obtained via its `deberta` attribute.
            Provides the hidden representation of each token within each sequence
            in the input batch.
        deberta_pooler : transformers.models.deberta_v2.modeling_deberta_v2.ContextPooler
            DeBERTa model pooler layer, obtained via its `pooler` attribute.
            Provides pooling over the latent representation of the tokens in each
            sequence, reducing over the sequence lenght dimension so one representation
            PER SEQUENCE is outputted.
        deberta_dropout : transformers.models.deberta_v2.modeling_deberta_v2.StableDropout
            DeBERTa model dropout layer, obtained via its `dropout` attribute.
            A dropout layer used between the pooler layer and the final classification head.
        num_labels : int
            Number of possible class labels.
        annotator_ids : list
            List of annotator IDs.
        """
        super().__init__(**kwargs)

        self.deberta_encoder = deberta_encoder
        self.deberta_pooler = deberta_pooler
        self.deberta_dropout = deberta_dropout

        self.num_labels = num_labels
        self.annotator_ids = annotator_ids

        # self.classification_heads = torch.nn.ModuleDict({
        #     str(annotator_id): torch.nn.Linear(
        #         in_features=768, out_features=2, bias=True
        #     ).to(device=self.deberta_encoder.device)
        #     for annotator_id in self.annotator_ids
        # })
        self.classification_heads = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=768, out_features=2, bias=True
            ).to(device=self.deberta_encoder.device)
            for _ in self.annotator_ids
        ])

        # Outdated: we're now accessing the classification heads by index
        # instead of by annotator ID.
        # try:
        #     assert set(self.annotator_ids) == set(range(len(self.classification_heads)))
        # except AssertionError:
        #     raise Exception('Annotator IDs must cover a full range')

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        annotator_ids=None
    ):
        """
        Adapted from: https://github.com/huggingface/transformers/blob/9d2056f12b66e64978f78a2dcb023f65b2be2108/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1155C5-L1166C6

        Everything remains as in the original implementation from `transformers`,
        but a list of annotator IDs of length `batch_size` must be provided, with
        the i-th annotator ID corresponding to the i-th sample in the batch.

        Example of usage:
        ```
        deberta_with_annotator_heads_model(
            **deberta_tokenizer(list_of_texts, padding=True, return_tensors='pt'),
            annotator_ids=list_of_ids
        )
        ```

        Notes:
          * The annotator-specific heads are accessed BY INDEX, with the
            indexing provided by the `annotator_ids` list provided to the
            model's constructor.

            Meaning: if we need to compute predicition for the annotator with
            ID `<ID1>`, we first look for the corresponding index in the
            `annotator_ids` list (`self.annotator_ids.index(<ID1>)`) and then
            we use such index to access the `ModuleList` containing the
            classification heads,
                    ID1 -> Index -> Classification head
        """
        # Standard DeBERTa forward pass to get sequences' latent
        # representations.
        outputs = self.deberta_encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.deberta_pooler(encoder_layer)
        pooled_output = self.deberta_dropout(pooled_output)

        # Loop over each (sample, annotator_id) pair, compute logits
        # with the classification head corresponding to the annotator
        # and concatenate all the logits into a single tensor.
        # Final shape: (batch_size, num_classes).
        # logits = torch.cat(
        #     [
        #         self.classification_heads[str(annotator_id.cpu().numpy())](pooled_output[i, ...])[None, ...]
        #         if not isinstance(annotator_id, int) else self.classification_heads[str(annotator_id)](pooled_output[i, ...])[None, ...]
        #         for i, annotator_id in enumerate(annotator_ids)
        #     ],
        #     dim=0
        # )
        logits = torch.cat(
            [
                self.classification_heads[
                    self.annotator_ids.index(annotator_id)
                ](pooled_output[i, ...])[None, ...]
                for i, annotator_id in enumerate(annotator_ids)
            ],
            dim=0
        )

        # Add computation of the loss if labels are passed?
        # See: https://huggingface.co/docs/transformers/custom_models
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


class DebertaWithAnnotatorHeadsPretrainedConfig(transformers.PretrainedConfig):
    def __init__(
        self,
        num_labels=2,
        annotator_ids=None,
        deberta_model_dir=None,
        **kwargs,
    ):
        """
        WARNING: apparently for the config to work all the arguments must
                 have a default value.
        """
        self.num_labels = num_labels
        self.annotator_ids = annotator_ids
        self.deberta_model_dir = deberta_model_dir

        super().__init__(
            num_labels=num_labels,
            annotator_ids=annotator_ids,
            deberta_model_dir=deberta_model_dir,
            **kwargs
        )


class DebertaWithAnnotatorHeadsPretrained(transformers.PreTrainedModel):
    """
    Same as the `DebertaWithAnnotatorHeads` model, but compliant with Hugging
    Face's Transformers, allowing for loading a checkpoint saved by the
    Tranformers library's `Trainer` object.

    Notes:
      * The correct config file (a 
        `DebertaWithAnnotatorHeadsPretrainedConfig` object) must have been
        saved in the same directory as the checkpoint that we're trying to
        load.
      * Provided the correct config object is there, this class can be used to
        load a checkpoint for the `DebertaWithAnnotatorHeads` (e.g. in case
        that model object was passed to the `Trainer`). Because
        `DebertaWithAnnotatorHeads` doesn't have its own config, an
        appropriate config must be created and placed in the checkpoint's
        directory, for doing which we essentially need to know the annotator
        IDs used for the training run - all other information in the config is
        basically always the same.
      * This class hasn't been tested for training: for now it's always ever
        been used for loading a checkpoint of a trained
        `DebertaWithAnnotatorHeads` object.
    
    To load a checkpoint:
    ```
    model = DebertaWithAnnotatorHeadsPretrained.from_pretrained(<checkpoint_dir>).to(device=<device>)

    model.eval()
    ```
    """
    config_class = DebertaWithAnnotatorHeadsPretrainedConfig
    
    def __init__(
        self,
        config
    ):
        """
        """
        super().__init__(config)

        deberta_model = transformers.DebertaV2ForSequenceClassification.from_pretrained(
            'microsoft/deberta-v3-base',
            num_labels=config.num_labels,
            cache_dir=config.deberta_model_dir
        )

        self.deberta_encoder = deberta_model.deberta
        self.deberta_pooler = deberta_model.pooler
        self.deberta_dropout = deberta_model.dropout

        self.num_labels = config.num_labels
        self.annotator_ids = config.annotator_ids

        self.classification_heads = torch.nn.ModuleList([
            torch.nn.Linear(
                in_features=768, out_features=2, bias=True
            ).to(device=self.deberta_encoder.device)
            for _ in self.annotator_ids
        ])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        annotator_ids=None
    ):
        """
        Adapted from: https://github.com/huggingface/transformers/blob/9d2056f12b66e64978f78a2dcb023f65b2be2108/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L1155C5-L1166C6

        Everything remains as in the original implementation from `transformers`,
        but a list of annotator IDs of length `batch_size` must be provided, with
        the i-th annotator ID corresponding to the i-th sample in the batch.

        Example of usage:
        ```
        deberta_with_annotator_heads_model(
            **deberta_tokenizer(list_of_texts, padding=True, return_tensors='pt'),
            annotator_ids=list_of_ids
        )
        ```

        Notes:
          * The annotator-specific heads are accessed BY INDEX, with the
            indexing provided by the `annotator_ids` list provided to the
            model's constructor.

            Meaning: if we need to compute predicition for the annotator with
            ID `<ID1>`, we first look for the corresponding index in the
            `annotator_ids` list (`self.annotator_ids.index(<ID1>)`) and then
            we use such index to access the `ModuleList` containing the
            classification heads,
                    ID1 -> Index -> Classification head
        """
        # Standard DeBERTa forward pass to get sequences' latent
        # representations.
        outputs = self.deberta_encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.deberta_pooler(encoder_layer)
        pooled_output = self.deberta_dropout(pooled_output)

        # Loop over each (sample, annotator_id) pair, compute logits
        # with the classification head corresponding to the annotator
        # and concatenate all the logits into a single tensor.
        # Final shape: (batch_size, num_classes).
        logits = torch.cat(
            [
                self.classification_heads[
                    self.annotator_ids.index(annotator_id)
                ](pooled_output[i, ...])[None, ...]
                for i, annotator_id in enumerate(annotator_ids)
            ],
            dim=0
        )

        # Add computation of the loss if labels are passed?
        # See: https://huggingface.co/docs/transformers/custom_models
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
