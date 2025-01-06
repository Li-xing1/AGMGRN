from torch.utils.data import Dataset, DataLoader
from utils.Data import *


class My_Dataset(Dataset):
    def __init__(self, cfgs, split):
        Update(cfgs)
        self.data = torch.load(
            f'data/Preprocessing-data-sets/{cfgs["name"]}/{cfgs["debug"]}/{cfgs["in_len"]}-{cfgs["out_len"]}/{split}.pt')
        if split == 'train':
            self.statics = self.data['statics']

    def __len__(self):
        return len(self.data['input_norm'])

    def __getitem__(self, item):
        return self.data['input_norm'][item] \
            , self.data['target_norm'][item] \
            , self.data['target_unnorm'][item] \
            , self.data['target_time_flag'][item] \
            , {'input_temporal_characterisrics': self.data['input_temporal_characterisrics'][item]
            , 'target_temporal_characterisrics': self.data['target_temporal_characterisrics'][item]}


if __name__ == '__main__':
    cfgs = yaml.safe_load(open(r'../cfgs/datasets\WHBT.yaml'))['dataset']
    cfgs['root'] = fr'D:\MGT\data\WHBT'
    data = My_Dataset(cfgs, 'train')
    dataloader = DataLoader(dataset=data, batch_size=16)
    for x, y, y1, target_time_flag, c in dataloader:
        print(y1.shape)
        print(target_time_flag.shape)
        break
