import pandas as pd
class Imbalanced(object):
    def __init__(self, df, attribute) -> None:
        self.column = df[attribute]
        self.hashmap = {}
        for item in self.column:
            if item in self.hashmap:
                self.hashmap[item] += 1
            else:
                self.hashmap[item] = 1
        
        self.equal_size = len(self.column) / len(set(self.column))
        
        self.min_value, self.min_counter = min(self.hashmap.items(), key=lambda x: x[1])
        self.second_min_counter = min(filter(lambda x : x != self.min_counter, self.hashmap.values()))
        self.max_value, self.max_counter = max(self.hashmap.items(), key=lambda x: x[1])
        self.second_max_counter = max(filter(lambda x : x != self.max_counter, self.hashmap.values()))
        
    
    def enum_check_min(self, tA, delta) -> bool:
        if tA == self.min_value:
            if self.max_value - self.hashmap[tA] < delta * self.equal_size:
                return False
            return True
        elif self.hashmap[tA] - self.min_counter < delta * self.equal_size:
            return False
        return True
    
    def enum_check_max(self, tA, delta) -> bool:
        if tA == self.max_value:
            if self.hashmap[tA] - self.min_value < delta * self.equal_size:
                return False
            return True
        elif self.max_counter - self.hashmap[tA] < delta * self.equal_size:
            return False
        return True
    
    def enum_check(self, tA, delta) -> bool:
        return self.enum_check_min(tA, delta) or self.enum_check_max(tA, delta)


if __name__ == "__main__":
    filename = "../kaggle_datasets/balita/data_balita.csv"
    attribute = "Nutrition_Status"
    df = pd.read_csv(filename)
    imbalanced = Imbalanced(df, attribute)
    tA = "normal"
    delta = 2
    print(imbalanced.enum_check(tA, delta))
    