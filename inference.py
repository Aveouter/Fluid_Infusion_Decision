import torch
import argparse
import data

def inference(model, data_loader, args):
    model.to(args.device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(args.device)
            water_out, gel_out, crystal_out, flow_out = model(inputs)

            predictions.append({
                "water_out": water_out.cpu(),
                "gel_out": gel_out.cpu(),
                "crystal_out": crystal_out.cpu(),
                "flow_out": flow_out.cpu()
            })

    return predictions