import pandas as pd
from bert_model_builder import BertClassificationModel, split_train_test

TEST_CLASSES = {"Software Engineer": 0, "Data Science": 1, "Data Analyst": 2, "DevOPS": 3, "Research": 4,
                "HW Engineer": 5, "QA, support": 6, "Product": 7, "Other": 8}

TRAINED_MODEL_PATH = "BERT-v1"

df = pd.read_csv('df_classes_formatted.csv')
text = df.work_position.values
labels = df.Tag.values


def test_train():
    model = BertClassificationModel(TEST_CLASSES, gpu=True)
    train_set, test_set = split_train_test(text, labels, model.tokenizer, test_ratio=0.2)
    model.train(train_set)
    model.test(test_set)


def test_predict():
    model = BertClassificationModel(TEST_CLASSES)
    model.load(TRAINED_MODEL_PATH)
    test_input = "Junior machine designer"
    assert model.predict(test_input) in TEST_CLASSES.keys()
