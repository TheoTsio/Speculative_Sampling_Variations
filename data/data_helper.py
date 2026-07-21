import pandas as pd

class Data():
    def __init__(self, split='train'):
        splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
        self.df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits[split])
        
        
if __name__ == "__main__":
    data = Data(split='train')
    print(data.df.head())