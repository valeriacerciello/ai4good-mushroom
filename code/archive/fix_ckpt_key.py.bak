"""Fix checkpoint key names so `fusion_clip_resnet.load_resnet_from_ckpt` can load state_dict.

If a checkpoint contains 'model_state_dict' but not 'state_dict', create a new file with 'state_dict'
key and copy labels (if present). Saves to the same path with suffix '.fixed.pth'.
"""
import torch
import sys
from pathlib import Path

def main(path):
    p = Path(path)
    sd = torch.load(p, map_location='cpu')
    if 'state_dict' in sd:
        print('Already has state_dict key; nothing to do')
        return
    if 'model_state_dict' in sd:
        new = {}
        new['state_dict'] = sd['model_state_dict']
        if 'labels' in sd:
            new['labels'] = sd['labels']
        out = p.with_suffix('.fixed.pth')
        torch.save(new, out)
        print('Wrote fixed checkpoint to', out)
    else:
        print('No model_state_dict key found; cannot fix automatically')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fix_ckpt_key.py <checkpoint.pth>')
    else:
        main(sys.argv[1])
