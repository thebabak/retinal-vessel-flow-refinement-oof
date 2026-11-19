"""Compilation, rendered QA and ZIP verification. Visual approval is explicit."""
from __future__ import annotations
import argparse,json,re,shutil,subprocess,zipfile,hashlib
from pathlib import Path
from .common import ROOT,load,write_json,stamp,sha256_file
PDFLATEX=Path('C:/Program Files/MiKTeX/miktex/bin/x64/pdflatex.exe')
RUNTIME=Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1116-1210-0_26.1116.1210.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe')

def compile_project(project,out):
    out.mkdir(exist_ok=True)
    for i in range(3):
        r=subprocess.run([str(PDFLATEX),'-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory',str(out),'main.tex'],cwd=project,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        (out/f'compile_pass_{i+1}.txt').write_text(r.stdout,encoding='utf-8');assert r.returncode==0,r.stdout[-5000:]
    log=(out/'main.log').read_text(encoding='utf-8',errors='replace');assert not re.search(r'(?:Reference|Citation) .+ undefined|There were undefined references|LaTeX Error',log)
    return out/'main.pdf',log

def prepare(state):
    project=Path(state['project']);out=Path(state['output']);tag=state['created_at'];tex=(project/'main.tex').read_text(encoding='utf-8')
    refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex));labels=re.findall(r'\\label\{([^}]+)\}',tex);assert not refs-set(labels),sorted(refs-set(labels));assert len(labels)==len(set(labels))
    citations={k.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',tex) for k in group.split(',')};bib=set(re.findall(r'\\bibitem\{([^}]+)\}',tex));assert not citations-bib
    assert Path(state['tex']).read_bytes()==(project/'main.tex').read_bytes()
    pdf,log=compile_project(project,out/'build');dest=ROOT/'output/pdf'/f'retinal_flow_all_datasets_{tag}.pdf';dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(pdf,dest)
    qa=out/f'PDF_QA_{tag}';qa.mkdir();script=qa/f'inspect_render_{tag}.py'
    script.write_text('''import json,re,sys
from pathlib import Path
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
pdf=Path(sys.argv[1]);qa=Path(sys.argv[2]);tag=sys.argv[3]
reader=PdfReader(pdf);texts=[p.extract_text() or '' for p in reader.pages]
assert all(texts) and not any('??' in s for s in texts)
(qa/f'PDF_text_{tag}.json').write_text(json.dumps(texts,ensure_ascii=False,indent=2),encoding='utf-8')
images=sorted(qa.glob(f'page_{tag}-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]));assert len(images)==len(texts)
sheets=[]
for start in range(0,len(images),12):
 sheet=Image.new('RGB',(1600,1800),'#dddddd');draw=ImageDraw.Draw(sheet)
 for j,path in enumerate(images[start:start+12]):
  im=Image.open(path).convert('RGB');im.thumbnail((390,560));x=(j%4)*400+(400-im.width)//2;y=(j//4)*600+25;sheet.paste(im,(x,y));draw.text(((j%4)*400+10,(j//4)*600+5),f'Page {start+j+1}',fill='black')
 path=qa/f'contact_{tag}_{start//12+1:02d}.png';sheet.save(path);sheets.append(str(path))
(qa/f'render_inventory_{tag}.json').write_text(json.dumps(dict(pages=len(texts),images=[str(p) for p in images],contact_sheets=sheets),indent=2),encoding='utf-8')
print(json.dumps(dict(pages=len(texts),contact_sheets=sheets)))
''',encoding='utf-8')
    poppler=shutil.which('pdftoppm');assert poppler
    subprocess.run([poppler,'-r','100','-png',str(dest),str(qa/f'page_{tag}')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    result=subprocess.run([str(RUNTIME),str(script),str(dest),str(qa),tag],check=True,capture_output=True,text=True);render=json.loads(result.stdout)
    write_json(out/f'compilation_checks_{tag}.json',dict(labels=len(labels),citations=len(citations),references=len(bib),PDF_pages=render['pages'],undefined_refs=False,source_SHA256=sha256_file(project/'main.tex'),PDF_SHA256=sha256_file(dest),overfull_boxes=re.findall(r'Overfull \\hbox[^\n]*',log),visual_review_required=True))
    state.update(pdf=str(dest),PDF_pages=render['pages'],render_QA=str(qa),contact_sheets=render['contact_sheets'],status='compiled_all_pages_rendered_requires_visual_review');write_json(out/f'BUILD_STATE_{tag}.json',state);write_json(ROOT/'tmp/current_all_dataset_MDPI_build.json',state);return state

def package(state):
    out=Path(state['output']);project=Path(state['project']);tag=state['created_at'];review=out/f'visual_review_{tag}.json';assert review.exists() and load(review)['all_pages_reviewed'] and load(review)['no_unresolved_layout_defects']
    # Preserve the previous paper/source hashes captured before the extension.
    run=Path(state['run']);protected=next(run.glob('prior_manuscripts_sources_protected_*.json'));original=load(protected);assert all(sha256_file(p)==h for p,h in original.items())
    now=stamp();archive=out/f'FINAL_MDPI_PDF_LaTeX_Overleaf_{now}.zip';mapping={p.relative_to(project).as_posix():p for p in project.rglob('*') if p.is_file()};mapping[Path(state['pdf']).name]=Path(state['pdf'])
    for p in (out/'evidence').rglob('*'):
        if p.is_file():mapping['evidence/'+p.relative_to(out/'evidence').as_posix()]=p
    for p in out.glob('OVERLEAF_INSTRUCTIONS_*.txt'):mapping[p.name]=p
    manifest={k:sha256_file(p) for k,p in mapping.items()};mp=out/f'package_manifest_{now}.json';write_json(mp,manifest);mapping[mp.name]=mp
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(mapping.items()):z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        extracted=out/f'ZIP_compile_check_{now}';extracted.mkdir();z.extractall(extracted)
    pdf,_=compile_project(extracted,extracted/'build');script=out/f'compare_ZIP_PDF_text_{now}.py';script.write_text("from pypdf import PdfReader\nimport sys\na=PdfReader(sys.argv[1]);b=PdfReader(sys.argv[2]);assert len(a.pages)==len(b.pages)\nassert all(x.extract_text()==y.extract_text() for x,y in zip(a.pages,b.pages))\nprint(len(a.pages))\n",encoding='utf-8');subprocess.run([str(RUNTIME),str(script),str(state['pdf']),str(pdf)],check=True)
    state.update(zip=str(archive),zip_SHA256=sha256_file(archive),zip_members=len(mapping),CRC_and_member_hashes_verified=True,extracted_ZIP_compilation_verified=True,all_previous_bound_files_unchanged=True,final_verified_at=now,status='completed_verified_MDPI_research_draft',github_updated=False)
    write_json(ROOT/'tmp/current_all_dataset_MDPI_delivery.json',state);write_json(out/f'FINAL_delivery_{now}.json',state);return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--package',action='store_true');p.add_argument('--state',type=Path);a=p.parse_args();state=load(a.state or ROOT/'tmp/current_all_dataset_MDPI_build.json')
    if a.prepare:print(json.dumps(prepare(state),indent=2))
    elif a.package:print(json.dumps(package(state),indent=2))
