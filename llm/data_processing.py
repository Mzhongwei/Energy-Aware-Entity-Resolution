import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
    
def sequence_grnerating(path):
    df = pd.read_csv(path)
    df_result = pd.DataFrame(columns=['text1', 'text2', 'labels'])
    for _, df_row in tqdm(df.iterrows(), total=len(df), desc="# Reading data"):
        text1 = ""
        text2 = ""
        labels = 0
        for col in df.columns:
                col_name_list = col.split('.')
                if col == 'label':
                    labels = df_row[col]
                elif len(col_name_list) > 1 and col_name_list[1] != "id": 
                    if int(col_name_list[0][-1]) == 1:
                        text1 = text1 + str(col_name_list[1]) + str(df_row[col])
                    elif int(col_name_list[0][-1]) == 2:
                        text2 = text2 + str(col_name_list[1]) + str(df_row[col])
        df_result.loc[len(df_result)] = [text1, text2, labels]
    return df_result

def preparing_training(trainset_path, validset_path):
    '''
    Load data from the dataset path and convert it to the correct format
    If the file size exceeds 1 GB, switch to other formats
    '''
    train_df = sequence_grnerating(trainset_path)
    trainset = Dataset.from_pandas(train_df)
    valid_df = sequence_grnerating(validset_path)
    validset = Dataset.from_pandas(valid_df)

    dataset = DatasetDict({
        "train": trainset,
        "eval": validset
    })

    return dataset

def preparing_testing(testset_path):
    test_df = sequence_grnerating(testset_path)
    testset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "test": testset
    })

    return dataset