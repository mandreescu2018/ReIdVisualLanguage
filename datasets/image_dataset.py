import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self._records = list(dataframe[['img_path','pid','camid','trackid']].itertuples(index=False, name=None))
            self.transform = transform

        def __len__(self):
            return len(self._records)

        def __getitem__(self, index):
            img_path, pid, camid, trackid = self._records[index]
            img = self.read_image(img_path)
            img = self.transform(img)
            return img, pid, camid, trackid
        
        @staticmethod
        def read_image(img_path, max_retries=10):
            """Keep reading image until succeed.
            This can avoid IOError incurred by heavy IO process."""
            if not os.path.exists(img_path):
                raise IOError(f"{img_path} does not exist")
            for attempt in range(max_retries):
                try:
                    return Image.open(img_path).convert('RGB')
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            raise IOError(f"Failed to read '{img_path}' after {max_retries} attempts")

