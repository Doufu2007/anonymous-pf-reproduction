from pathlib import Path
import hashlib,json,sys
here=Path(__file__).resolve().parent
source=here/'test-red.log'
original=source.read_text(encoding='utf-8-sig')
revised=original.replace(str(here),'<revision-qgpf>')
revised=revised.replace(str(here.parents[2]),'<repo>')
revised=revised.replace(sys.base_prefix,'<python-runtime>')
target=here/'test-red-public.log'
target.write_text(revised,encoding='utf-8')
manifest={'schema':'revision.qgpf.public-log-conversion.v1',
'source_local_only':'code/revision_20261009/qgpf/test-red.log',
'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
'public_copy':'code/revision_20261009/qgpf/test-red-public.log',
'public_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
'transformation':'Absolute revision/repository and Python-runtime path prefixes replaced with neutral tokens; original source bytes not changed',
'original_must_be_excluded_from_publication':True}
(here/'public-log-conversion.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
