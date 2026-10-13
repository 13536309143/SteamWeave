"""Fetch only the official GloVe 6B 100d member using byte ranges."""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path
import httpx
import numpy as np
from sg_agents.config import ARTIFACTS
from sg_agents.data import load_dataset

URL='https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'

def tokenize(text):return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+",text.lower())

class RemoteZip(io.RawIOBase):
    def __init__(self,url,block=8*1024*1024):
        self.client=httpx.Client(timeout=120,follow_redirects=True)
        self.url=url;self.block=block;self.pos=0;self.cache={};self.downloaded=0
        r=self.client.get(url,headers={'Range':'bytes=-131072'})
        if r.status_code!=206:raise RuntimeError('Official GloVe server must support Range requests')
        self.size=int(r.headers['content-range'].split('/')[-1])
        self.tail_start=self.size-len(r.content);self.tail=r.content
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos);parts=[]
        while n>0:
            if self.pos>=self.tail_start:
                take=min(n,self.size-self.pos);part=self.tail[self.pos-self.tail_start:self.pos-self.tail_start+take]
            else:
                start=self.pos//self.block*self.block
                if start not in self.cache:
                    end=min(self.size-1,start+self.block-1)
                    for attempt in range(3):
                        try:
                            r=self.client.get(self.url,headers={'Range':f'bytes={start}-{end}'})
                            if r.status_code!=206 or not r.headers.get('content-range','').startswith(f'bytes {start}-'):
                                raise RuntimeError('GloVe Range response does not match requested bytes')
                            self.cache={start:r.content};self.downloaded+=len(r.content)
                            if self.downloaded//(32*1024*1024)!=(self.downloaded-len(r.content))//(32*1024*1024):
                                print(f'GloVe compressed data fetched: {self.downloaded/1024/1024:.0f} MiB',flush=True)
                            break
                        except (httpx.HTTPError,RuntimeError):
                            if attempt==2:raise
                off=self.pos-start;take=min(n,len(self.cache[start])-off);part=self.cache[start][off:off+take]
            if not part:break
            parts.append(part);self.pos+=len(part);n-=len(part)
        return b''.join(parts)
    def close(self):self.client.close();super().close()

def prepare():
    destination=ARTIFACTS/'research'/'glove_vocab.npz';destination.parent.mkdir(exist_ok=True,parents=True)
    vocabulary=sorted({w for r in load_dataset() for w in tokenize(r.text)})
    if destination.exists():
        old=np.load(destination)
        if old['words'].tolist()==vocabulary:
            print('GloVe vocabulary cache ready.',flush=True);return destination
    bundled=Path(__file__).resolve().parent/'assets'/'glove_vocab.json'
    if bundled.exists():
        subset=json.loads(bundled.read_text())
        if subset['words']==vocabulary:
            np.savez_compressed(destination,words=np.array(subset['words']),vectors=np.array(subset['vectors'],dtype=np.float32),found=np.array(subset['found'],dtype=bool))
            print('Restored official GloVe subset from the bundled research asset.',flush=True)
            return destination
    wanted=set(vocabulary);found={}
    print(f'Preparing official 100d GloVe vectors for {len(wanted)} corpus words.',flush=True)
    with RemoteZip(URL) as remote:
        with zipfile.ZipFile(remote) as z:
            print('Official ZIP entries:',z.namelist(),flush=True)
            with z.open('glove.6B.100d.txt') as member:
                for line in io.TextIOWrapper(member,encoding='utf-8'):
                    word,_,rest=line.rstrip().partition(' ')
                    if word in wanted:
                        vector=np.fromstring(rest,sep=' ',dtype=np.float32)
                        if len(vector)==100:found[word]=vector
            fetched=remote.downloaded
    vectors=np.stack([found.get(w,np.zeros(100,dtype=np.float32)) for w in vocabulary])
    np.savez_compressed(destination,words=np.array(vocabulary),vectors=vectors,found=np.array([w in found for w in vocabulary]))
    metadata={'source':URL,'member':'glove.6B.100d.txt','embedding_dim':100,
              'corpus_vocabulary':len(vocabulary),'covered_words':len(found),
              'missing_words':[w for w in vocabulary if w not in found],
              'fetched_compressed_bytes':fetched,'note':'Fixed pretrained lexical vectors; no labels used.'}
    destination.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True);return destination

if __name__=='__main__':prepare()
