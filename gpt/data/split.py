class Split:
    def __init__(self, train_pct, val_pct, test_pct=0):
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
    
    def split_it(self, data):
        n = int(self.train_pct*len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data