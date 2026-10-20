import hashlib, json, subprocess
from pathlib import Path
import yaml
root=Path('C:/Repos/EcuadorElectricGrid-environment-validation')
m=yaml.safe_load((root/'config/artifacts.yaml').read_text())
for a in m['artifacts']:
    if a['id'] not in ['generation_existing_2022_source','generation_location_evidence_2022']: continue
    b=subprocess.check_output(['git','show','HEAD:'+a['destination']],cwd=root)
    lf=b.replace(b'\r\n',b'\n')
    versions={'git':b,'CRLF':lf.replace(b'\n',b'\r\n'),'git_BOM':b'\xef\xbb\xbf'+b,'CRLF_BOM':b'\xef\xbb\xbf'+lf.replace(b'\n',b'\r\n'),'working':(Path('C:/Repos/EcuadorElectricGrid')/a['destination']).read_bytes()}
    print(a['id'], 'expected',a['bytes'],a['sha256'])
    for k,v in versions.items(): print(k,len(v),hashlib.sha256(v).hexdigest(),repr(v[:50]))
