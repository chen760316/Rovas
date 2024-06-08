import pandas as pd
import time

class SDomian(object):
    def __init__(self, filename, attribute) -> None:
        self.column = pd.read_csv(filename)[attribute]
        self.count = len(set(self.column))
    
    def enum_check(self, sigma) -> bool:
        if self.count < sigma:
            return True
        return False

if __name__ == "__main__":
    filename = "../kaggle_datasets/balita/data_balita.csv"
    attribute = "Nutrition_Status"
    start_time = time.time()
    imbalanced = SDomian(filename, attribute)
    sigma = 2
    print(imbalanced.enum_check(sigma))
    end_time = time.time() 
    print(f"{end_time - start_time}s")