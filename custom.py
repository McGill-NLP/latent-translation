from typing import Optional, Iterable, Callable, List, Union, Tuple
from collections import UserDict
import logging
import torch
from torch import nn
from torch.nn import functional as F
from transformers import MBartConfig, MBartModel, M2M100Config, M2M100Model
from transformers.models.mbart.modeling_mbart import MBartPreTrainedModel, MBartClassificationHead
from transformers.models.m2m_100.modeling_m2m_100 import M2M100PreTrainedModel
from transformers.generation_utils import (
    GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput,
)
from transformers.generation_beam_search import BeamScorer, BeamHypotheses
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


class MBartForMultipleChoice(MBartPreTrainedModel):
    def __init__(self, config: MBartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = MBartModel(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            1,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=flat_inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = flat_input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return CustomModelOutput(
            loss=loss,
            logits=reshaped_logits,
        )


class M2M100ForMultipleChoice(M2M100PreTrainedModel):
    def __init__(self, config: M2M100Config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = M2M100Model(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            1,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=flat_inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            )
        hidden_states = outputs.last_hidden_state  # last hidden state

        eos_mask = flat_input_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output


        return CustomModelOutput(
            loss=loss,
            logits=reshaped_logits,
        )


class CustomModelOutput(object):
    def __init__(self, loss=None, logits=None, nmt_loss=None,
                 all_logits=None, translations=None, scores=None):
        self.loss = loss
        self.logits = logits
        self.nmt_loss = nmt_loss
        self.all_logits = all_logits
        self.translations = translations
        self.scores = scores


class TranslatorClassifier(nn.Module):
    def __init__(self, classifier, translator, cls_tokenizer, nmt_tokenizer,
                 num_samples=1, do_sample=False, max_length=250, reranker=None,
                 forced_bos_token_id=None, train_translator=False, self_training=False):
        super().__init__()
        self.classifier = classifier
        self.translator = translator
        for param in self.translator.parameters():
            param.requires_grad = train_translator
        self.train_translator = train_translator
        self.cls_tokenizer = cls_tokenizer
        self.nmt_tokenizer = nmt_tokenizer
        self.num_samples = num_samples
        self.do_sample = do_sample
        self.max_length = max_length
        self.reranker = reranker
        self.forced_bos_token_id = forced_bos_token_id
        self.self_training = self_training

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                num_beams=12, temperature=1.):

        batch_size, num_sent, seq_len = input_ids.shape
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        if self.train_translator and self.training:
            generate_function = self.translator.generate_with_grad
            temperature = 1.
        else:
            generate_function = self.translator.generate

        nmt_out = generate_function(input_ids=input_ids, attention_mask=attention_mask,
                                    do_sample=self.do_sample, num_beams=num_beams,
                                    num_return_sequences=self.num_samples,
                                    temperature=temperature,
                                    forced_bos_token_id=self.forced_bos_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    max_length=self.max_length,
                                    )
        nmt_text = [self.nmt_tokenizer.decode(t, skip_special_tokens=True) for t in nmt_out.sequences]
        rearranged_nmt_text, new_input_ids, new_attention_mask = self.rearrange(nmt_text, batch_size, num_sent, input_ids.device)

        cls_out = self.classifier(input_ids=new_input_ids, attention_mask=new_attention_mask)
        logits = cls_out.logits.view(batch_size, self.num_samples, -1)
        if self.reranker is None:
            logits = logits.mean(1)
        else:
            # new_input_ids : (batch_size * num_samples x num_choices x seq_len) OR (batch_size * num_samples x seq_len)
            if len(new_input_ids.shape) == 2:
                input_ids_to_rerank = new_input_ids.view(batch_size, self.num_samples, seq_len)
                attention_mask_to_rerank = new_attention_mask.view(batch_size, self.num_samples, seq_len)
            elif len(new_input_ids.shape) == 3:
                num_choices = new_input_ids.shape[1]
                input_ids_to_rerank = new_input_ids.view(batch_size, self.num_samples, num_choices, seq_len).transpose(1, 2).reshape(batch_size * num_choices, self.num_samples, seq_len)
                attention_mask_to_rerank = new_attention_mask.view(batch_size, self.num_samples, num_choices, seq_len).transpose(1, 2).reshape(batch_size * num_choices, self.num_samples, seq_len)
            else:
                raise ValueError
            reranker_outputs = self.reranker(input_ids=input_ids_to_rerank, attention_mask=attention_mask_to_rerank)
            score_weights = F.softmax(reranker_outputs.logits, dim=-1).unsqueeze(-1)
            logits = (logits * score_weights).sum(1)

        loss, nmt_loss = None, None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))

            if self.train_translator and self.training:
                nmt_loss = self.minimum_risk_training(nmt_out, cls_out, labels, num_sent)

        return CustomModelOutput(loss=loss, logits=logits, nmt_loss=nmt_loss,
                                 all_logits=cls_out.logits.view(batch_size, self.num_samples, -1),
                                 translations=rearranged_nmt_text,
                                 scores=nmt_out.sequences_scores.view(batch_size, num_sent, self.num_samples).sum(dim=1),
                                 )

    def minimum_risk_training(self, nmt_out, cls_out, labels, num_sent,
                              normalize=True,
                              subtract_baseline=False,
                              metric_reward=False,
                              ):
        batch_size = len(labels)

        # rewards
        with torch.no_grad():
            if metric_reward:
                sample_logits = cls_out.logits.view(batch_size, self.num_samples, -1)
                _, preds = torch.max(sample_logits, dim=-1)
                rewards = (preds == labels.unsqueeze(-1)).float()  # batch_size x num_samples
            else:
                sample_logits = cls_out.logits.view(batch_size * self.num_samples, -1)
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(sample_logits, labels.view(-1).repeat_interleave(self.num_samples))
                rewards = 1. - loss.view(batch_size, self.num_samples)

        log_prob = nmt_out.sequences_scores
        log_prob = log_prob.view(batch_size, num_sent, self.num_samples).sum(dim=1)  # batch_size x num_samples

        # normalize and subtract baseline
        if not self.self_training:
            if normalize:
                log_prob = F.softmax(log_prob, dim=1)
            if subtract_baseline:
                rewards = rewards - torch.mean(rewards, dim=1, keepdim=True)
            loss = (log_prob * - rewards).sum(1).mean(0)
        else:
            loss = log_prob[:, 0].sum()
        return loss

    def rearrange(self, nmt_text, batch_size, num_sent, device):
        rearranged_nmt_text = []
        cls_ids = []
        for i in range(0, batch_size * num_sent * self.num_samples, num_sent * self.num_samples):
            sample_text = []
            for j in range(self.num_samples):
                if num_sent == 2:
                    cls_id = self.cls_tokenizer.encode_plus(nmt_text[i + j], nmt_text[i + j + self.num_samples],
                                                            add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True)
                    cls_ids.append(cls_id)
                    sample_text.append([nmt_text[i + j], nmt_text[i + j + self.num_samples]])
                elif num_sent >= 4:
                    choice_ids = []
                    text_a = (nmt_text[i + j] + " " + nmt_text[i + j + self.num_samples]).strip()
                    choice_text = [nmt_text[i + j], nmt_text[i + j + self.num_samples]]
                    for k in range(2, num_sent):
                        text_b = nmt_text[i + j + (self.num_samples * k)]
                        cls_id = self.cls_tokenizer.encode_plus(text_a, text_b,
                                                                add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True)
                        choice_ids.append(cls_id)
                        choice_text.append(text_b)
                    cls_ids.append(choice_ids)
                    sample_text.append(choice_text)
                else:
                    raise ValueError
            rearranged_nmt_text.append(sample_text)

        if num_sent == 2:
            new_input_ids = torch.tensor([f.input_ids for f in cls_ids], dtype=torch.long, device=device)
            new_attention_mask = torch.tensor([f.attention_mask for f in cls_ids], dtype=torch.long, device=device)
        elif num_sent >= 4:
            new_input_ids = torch.tensor([[f.input_ids for f in choices] for choices in cls_ids], dtype=torch.long, device=device)
            new_attention_mask = torch.tensor([[f.attention_mask for f in choices] for choices in cls_ids], dtype=torch.long, device=device)

        return rearranged_nmt_text, new_input_ids, new_attention_mask


class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[bool] = False,
        num_beam_hyps_to_keep: Optional[int] = 1,
        num_beam_groups: Optional[int] = 1,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
        self.num_beam_groups = num_beam_groups
        self.group_size = self.num_beams // self.num_beam_groups

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

        if not isinstance(num_beam_groups, int) or (num_beam_groups > num_beams) or (num_beams % num_beam_groups != 0):
            raise ValueError(
                f"`num_beam_groups` has to be an integer smaller or equal than `num_beams` and `num_beams` "
                f"has to be divisible by `num_beam_groups`, but is {num_beam_groups} with `num_beams` being {num_beams}."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.group_size)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.group_size + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.group_size
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.group_size:
                    break

            if beam_idx < self.group_size:
                raise ValueError(
                    f"At most {self.group_size} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor]:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx]
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []
        best_scores = torch.zeros(batch_size * self.num_beam_hyps_to_keep, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp_tuple = sorted_hyps.pop()
                best_score = best_hyp_tuple[0]
                best_hyp = best_hyp_tuple[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)

                # append to lists
                best.append(best_hyp)
                best_scores[i * self.num_beam_hyps_to_keep + j] = best_score

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return UserDict(
            {
                "sequences": decoded,
                "sequence_scores": best_scores,
            }
        )



def generate_with_grad(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    encoder_no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    max_time: Optional[float] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    num_beam_groups: Optional[int] = None,
    diversity_penalty: Optional[float] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    forced_bos_token_id: Optional[int] = None,
    forced_eos_token_id: Optional[int] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:

    # set init values
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
    max_length = max_length if max_length is not None else self.config.max_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )

    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    model_kwargs["output_attentions"] = output_attentions
    model_kwargs["output_hidden_states"] = output_hidden_states

    if input_ids is None:
        # init `input_ids` with bos_token_id
        input_ids = self._prepare_input_ids_for_generation(bos_token_id, model_kwargs.get("encoder_outputs"))

    if model_kwargs.get("attention_mask", None) is None:
        # init `attention_mask` depending on `pad_token_id`
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            input_ids, pad_token_id, eos_token_id
        )

    # special case if pad_token_id is not defined
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        pad_token_id = eos_token_id

    # Storing encoder_input_ids for logits_processor that could use them
    encoder_input_ids = input_ids if self.config.is_encoder_decoder else None

    if self.config.is_encoder_decoder:
        # add encoder_outputs to model_kwargs
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(input_ids, model_kwargs)

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                input_ids, decoder_start_token_id=decoder_start_token_id, bos_token_id=bos_token_id
            )

        if "encoder_outputs" not in model_kwargs or not isinstance(model_kwargs["encoder_outputs"], ModelOutput):
            raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

    if input_ids.shape[-1] >= max_length:
        input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
            "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
        )

    # determine generation mode
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
    is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
    is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
    if num_beam_groups > num_beams:
        raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    if is_group_beam_gen_mode and do_sample is True:
        raise ValueError(
            "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
        )

    # set model_kwargs
    model_kwargs["use_cache"] = use_cache

    # get distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
        encoder_input_ids=encoder_input_ids,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        max_length=max_length,
        eos_token_id=eos_token_id,
        forced_bos_token_id=forced_bos_token_id,
        forced_eos_token_id=forced_eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
    )

    stopping_criteria = self._get_stopping_criteria(
        max_length=max_length,
        max_time=max_time,
    )

    if is_greedy_gen_mode:
        raise NotImplementedError

    elif is_sample_gen_mode:
        raise NotImplementedError

    elif is_beam_gen_mode:
        batch_size = input_ids.shape[0]

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )
        # interleave with `num_beams`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
        )
        return self.beam_search(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_beam_sample_gen_mode:
        logits_warper = self._get_logits_warper(
            top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
        )

        batch_size = input_ids.shape[0] * num_return_sequences

        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=self.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        # interleave with `num_beams * num_return_sequences`
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_beams * num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        return self.beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **model_kwargs,
        )

    elif is_group_beam_gen_mode:
        raise NotImplementedError
