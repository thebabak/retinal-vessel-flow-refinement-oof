"""Print saved paper values; never fit models or recompute metrics/statistics."""
from pathlib import Path
import json

def main():
    source=Path(__file__).resolve().parents[1]/'results/verified_paper_results.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    print('Saved raw mean Dice (sample SD over five fitted refiners)')
    print('Dataset       Baseline      O1 thin-direct       O2 endpoint-flow     O3 native-direct')
    for name,d in data['datasets'].items():
        columns=[]
        for arm in ('direct_thin_anchor','esfm_endpoint_anchor','native_patch_direct'):
            raw=d['options'][arm]['raw']
            columns.append(f"{raw['mean']['dice']:.6f} ({raw['sample_sd']['dice']:.6f})")
        print(f"{name:<12} {d['baseline']['dice']:.6f}     "+'   '.join(columns))
    print('One frozen baseline per dataset; comparative grid512. Intervals are exploratory.')

if __name__=='__main__':main()
