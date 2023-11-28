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


    # Manual loop through the dataloader for prediction
    for i in range(len(test_dataset)):
        for j, batch in enumerate(test_dataloader):
            if j == i:
                with torch.no_grad():
                    prediction = model(batch['feature'])

                # Update dataset with the latest prediction
            
                test_dataset.update_with_prediction(batch['pred_target'], prediction)

                # Break the loop after one prediction as you want to predict step-by-step
                break

        # # Recreate DataLoader with the updated dataset
        test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    result_df = test_dataset.df[['id', 'sbi']].iloc[7:]
    result_df.to_csv(f'outputs/{test_dataset.split}_{args.station_id}.csv', index=False)

    


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
