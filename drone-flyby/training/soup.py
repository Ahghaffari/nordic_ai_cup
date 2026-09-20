"""Average the weights of several checkpoints (a model soup) into one checkpoint."""
import sys
from copy import deepcopy
import torch

out, paths = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
base = deepcopy(cks[0])
sds = [ (c['ema'] or c['model']).float().state_dict() for c in cks]
avg = {}
for k, v in sds[0].items():
    avg[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype) if v.is_floating_point() else v
model = (base['ema'] or base['model']).float()
model.load_state_dict(avg)
base['ema'] = model.half()
base['model'] = None
base['train_args'] = dict(base['train_args'], name='soup:' + '+'.join(p.split('/')[-1].replace('.pt', '') for p in paths))
torch.save(base, out)
print('saved', out, 'from', len(paths))
