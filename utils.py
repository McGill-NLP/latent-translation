import copy
import json
import logging
import os
from tqdm import tqdm

from transformers import DataProcessor
from transformers.data.processors.squad import SquadExample

logger = logging.getLogger(__name__)


class MultiplicativeAnnealer(object):
    """Decays the temperature by gamma every epoch.

    Args:
        gamma (float): Multiplicative factor of temperature decay.
    """

    def __init__(self, gamma, temperature):
        self.gamma = gamma
        self.temperature = temperature

    def step(self):
        self.temperature *= self.gamma


class SCExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None, language=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.language = language

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class MCExample(object):
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, question, contexts, endings, label=None, language=None):
        self.guid = guid
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.language = language

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually    ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    preprocess_trans=False,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if not preprocess_trans:
            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True,
                                           max_length=max_length, pad_to_max_length=True)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            token_type_ids = inputs.get("token_type_ids", None)
        else:
            inputs = [tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
                      for text in [example.text_a, example.text_b]]
            input_ids = [x["input_ids"] for x in inputs]
            attention_mask = (
                [x["attention_mask"] for x in inputs] if "attention_mask" in inputs[0] else None
            )
            token_type_ids = (
                [x["token_type_ids"] for x in inputs] if "token_type_ids" in inputs[0] else None
            )
        label = label_map[example.label]

        if ex_index < 5 and not preprocess_trans:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            if token_type_ids is not None:
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features


def convert_examples_MC_to_features(
    examples,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    preprocess_trans=False
):
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        choices_inputs = []
        if not preprocess_trans:
            for text_b in example.endings:
                text_a = (example.contexts  + " " + example.question).strip()
                inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
                choices_inputs.append(inputs)
        else:
            for text_a in [example.contexts, example.question] + example.endings:
                inputs = tokenizer.encode_plus(text_a, add_special_tokens=True, max_length=max_length, pad_to_max_length=True)
                choices_inputs.append(inputs)
        label = example.label

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            for n_c in range(len(input_ids)):
                logger.info("Choice {}".format(n_c))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids[n_c]]))
                logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids[n_c])))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask[n_c]]))
                if token_type_ids is not None:
                    logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids[n_c]]))
            logger.info("label: %d (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    return features


class TiDiQAProcessor(DataProcessor):
    """
    Processor for the TyDiQA data set.
    """

    def get_examples(self, data_dir, language='en', set_type="train"):
        filename = f"tydiqa.{language}.{set_type}.json"
        with open(
            os.path.join(data_dir, filename), "r", encoding="utf-8-sig"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, set_type)

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry.get("title", "")
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        is_impossible=is_impossible,
                        answers=answers,
                        title=title,
                    )
                    examples.append(example)
        return examples
