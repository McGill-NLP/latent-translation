import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class RobertaForReranking(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * config.num_sent, 1)
        self.num_sent = config.num_sent
        self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_samples, _ = input_ids.shape if input_ids is not None else inputs_embeds.shape
        true_batch_size = batch_size // self.num_sent

        flat_input_ids = input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.reshape(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.reshape(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        new_output = pooled_output.view(true_batch_size, self.num_sent, num_samples, self.hidden_size).transpose(1, 2)
        new_output = new_output.reshape(true_batch_size * num_samples, self.num_sent * self.hidden_size)
        # (batch_size * num_samples, hidden_size * num_sent)
        logits = self.classifier(new_output)
        reshaped_logits = logits.view(-1, num_samples)

        loss = None

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
