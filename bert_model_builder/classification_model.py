import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import trange

from bert_model_builder.performance_analysis import b_metrics


class BertClassificationModel:

    def __init__(self, classes_mapped, gpu=False):
        self.classes_mapped = classes_mapped
        self._reverse_classes_mapping = {val: key for key, val in self.classes_mapped.items()}
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True
        )
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(self.classes_mapped.keys()),
            output_attentions=False,
            output_hidden_states=False,
        )
        if gpu:
            self.model.cuda()
        # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, eps=1e-08)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._trained = False

    def train(self, train_set, batch_size=16, epochs=2):
        """
        :param batch_size: Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
        :param epochs: Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
        """
        # Prepare DataLoader
        train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)

        for _ in trange(epochs, desc='Epoch'):
            # Set model to training mode
            self.model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids, token_type_ids=None,
                                          attention_mask=b_input_mask, labels=b_labels)
                # Backward pass
                train_output.loss.backward()
                self.optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        self._trained = True

    def test(self, test_set, batch_size=16):
        if not self._trained:
            raise Exception("Model must be trained first!")
        """
        :param batch_size: Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
        """
        test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)

        # Set model to evaluation mode
        self.model.eval()

        # Tracking variables
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in test_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                eval_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy) / len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision) / len(val_precision)) if len(
            val_precision) > 0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall) / len(val_recall)) if len(
            val_recall) > 0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity) / len(val_specificity)) if len(
            val_specificity) > 0 else '\t - Validation Specificity: NaN')

    def predict(self, input_txt):
        if not self._trained:
            raise Exception("Model must be trained first!")

        self.model.eval()

        # We need Token IDs and Attention Mask for inference on the new sentence
        test_ids = []
        test_attention_mask = []

        # Apply the tokenizer
        encoding = preprocessing(input_txt, self.tokenizer)

        # Extract IDs and Attention Mask
        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        # Forward pass, calculate logit predictions
        with torch.no_grad():
            output = self.model(test_ids.to(self.device), token_type_ids=None,
                                attention_mask=test_attention_mask.to(self.device))

        return self._reverse_classes_mapping[np.argmax(output.logits.cpu().numpy()).flatten().item()]

    def save(self, path):
        if not self._trained:
            raise Exception("Model must be trained first!")
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
        self._trained = True


def split_train_test(text, labels, tokenizer, test_ratio=0.2):
    """
    Split a given labeled dataset to train and test groups.
    :param test_ratio: % of data that will be used in test (1-test_ratio will be used for train)
    :return: tuple(train_set, test_set)
    """
    token_id = []
    attention_masks = []

    for sample in text:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=test_ratio,
        shuffle=True,
        stratify=labels)

    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels[train_idx])
    test_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels[val_idx])

    return train_set, test_set


def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
      - input_ids: list of token ids
      - token_type_ids: list of token type ids
      - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=32,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')