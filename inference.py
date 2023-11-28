from dataset import BikeDataset
from train import BikePredModule
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
import argparse
import pandas as pd

def main(args):
    # load checkpoint and hyperparameter
    ckpt = torch.load(args.ckpt_dir, map_location=lambda storage, loc: storage)
    model = BikePredModule.load_from_checkpoint(
        args.ckpt_dir,
        hparams=ckpt["hyper_parameters"],
    )

    # dataset and dataloader
    test_dataset = BikeDataset(args.station_id, split='test_1021', mode=model.hparams.mode)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    # trainer
    trainer = Trainer()

    # output
    predictions = [float(item) for sublist in trainer.predict(model, dataloaders=test_dataloader) for item in sublist]
    ids = [item for sublist in [batch['id'] for batch in test_dataloader] for item in sublist]

    print(ids)
    print(len(ids))
    print(predictions)
    print(len(predictions))

    # visualization
    dict = {'id': ids, 'prediction': predictions}
    df = pd.DataFrame(dict)
    df.to_csv(f'{args.ckpt_dir.split(".")[0]}_output.csv', index=False)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--station_id',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    main(args)
