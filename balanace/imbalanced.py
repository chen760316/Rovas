import pandas as pd
class Imbalanced(object):
    def __init__(self, filename, attribute) -> None:
        self.column = pd.read_csv(filename)[attribute]
        self.hashmap = {}
        for item in self.column:
            if item in self.hashmap:
                self.hashmap[item] += 1
            else:
                self.hashmap[item] = 1
        
        self.min_value, self.min_counter = min(self.hashmap.items(), key=lambda x: x[1])
        self.second_min_counter = min([x for x in self.hashmap.values() if x != self.min_counter]) 
        
    
    def enum_check(self, tA, delta) -> bool:
        if tA == self.min_value:
            if self.second_min_counter - self.hashmap[tA] < delta:
                return True
            return False
        elif self.min_counter - self.hashmap[tA] < delta:
            return True
        return False


if __name__ == "__main__":
    filename = "../kaggle_datasets/balita/data_balita.csv"
    attribute = "Nutrition_Status"
    imbalanced = Imbalanced(filename, attribute)
    tA = "normal"
    delta = 2
    print(imbalanced.enum_check(tA, delta))
    