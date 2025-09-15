import torch
import torch.nn as nn

from typing import (
    Optional, 
    Tuple,    
    Union
)

from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel

class EncoderWithAnnotatorHeads(torch.nn.Module):
    def __init__(
            self, 
            checkpoint, 
            num_annotators,
            label_weights=None, 
            num_labels=2,
            use_return_dict=True,
        ):
        super().__init__()
        self.num_labels = num_labels
        self.use_return_dict = use_return_dict

        self.pretrained_model = AutoModel.from_pretrained(checkpoint)
        self.dim = self.pretrained_model.config.hidden_size
        embedding_dim = self.dim * num_labels
        self.annotator_weights = nn.Embedding(num_annotators, embedding_dim)
        self.label_weights = label_weights

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        annotator_indecies: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        pretrained_output = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # using the first token's last hidden state like DistilBertForSequenceClassification
        # but could be pooled / averaged / ...
        text_vectors = pretrained_output.last_hidden_state[:, 0] # batch_size x 768
        annotator_embs = self.annotator_weights(annotator_indecies)  # num_annotators x (768 * num_labels)
        batch_size = input_ids.shape[0]
        annotator_weights = annotator_embs.view((batch_size, self.dim, self.num_labels))
        logits = torch.matmul(text_vectors.unsqueeze(-2), annotator_weights).squeeze() # batch_size x num_labels

        loss = None
        if labels is not None:
            if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                loss_fct = CrossEntropyLoss(
                    weight=self.label_weights
                )
                loss = loss_fct(logits, labels)
            else:
                raise NotImplementedError('Only supports "single_label_classification"')

        if not return_dict:
            output = (logits,) + pretrained_output[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=pretrained_output.hidden_states,
            attentions=pretrained_output.attentions
        )
    
if __name__ == "__main__":

    model = EncoderWithAnnotatorHeads(
        'distilbert/distilbert-base-cased',
        num_annotators=7
    )
    
    input_ids = torch.LongTensor([
        [1, 2, 3, 4, 0, 0, 0], 
        [6, 2, 5, 7, 134, 1, 1], 
        [1, 34, 113, 24, 44, 0, 0]
    ])

    annotator_indecies = torch.LongTensor([
        0, 
        2, 
        5
    ])

    labels = torch.LongTensor([
        0,
        1, 
        1
    ])


    print(model.annotator_weights(torch.LongTensor([2])))
    print(model.annotator_weights(torch.LongTensor([1])))

    out = model(
        input_ids=input_ids, 
        annotator_indecies=annotator_indecies,
        labels=labels
    )
    out.loss.backward()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optim.step()

    print(model.annotator_weights(torch.LongTensor([2])))
    print(model.annotator_weights(torch.LongTensor([1])))
