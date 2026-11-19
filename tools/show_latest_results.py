"""Display stored scalar values only. No scientific imports or recalculation."""
from pathlib import Path
import csv,json
def main():
    root=Path(__file__).resolve().parents[1]
    study=json.loads((root/'provenance/LATEST_STUDY.json').read_text(encoding='utf-8'))
    print(study['title'])
    print('Saved fresh-CV Dice means and SD (shared upstream fits; five correction seeds)')
    with (root/study['results']/study['mean_SD']).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['cohort'] in ('fresh_nested_CV','fresh_nested_CV_calibration') and r['phase'] in ('fresh_cv','calibration') and r['metric']=='dice' and r['output']=='selected':
                print(r['cohort'],r['dataset'],r['mode'],'mean='+r['mean'],'SD='+r['SD'])
    print('Selected fixed-threshold outputs preserve O3. A Flow-specific advantage is not established.')
    print('DRIVE local test-label provenance remains unverified. See the study documentation.')
if __name__=='__main__':main()
