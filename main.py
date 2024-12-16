if __name__ == '__main__':

    "csv path"
    path = "/hpc2hdd/home/yhuang489/OpenVid/data/train/OpenVid-1M.csv"
    import pandas as pd 
    data = pd.read_csv(path)
    for i, name in enumerate(data.video):
        if  "magictime" in name:
            print(data.iloc[i])
            break
