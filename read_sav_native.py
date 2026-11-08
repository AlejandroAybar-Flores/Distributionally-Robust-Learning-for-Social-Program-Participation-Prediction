import struct
from pathlib import Path


def inspect_sav(path):
    with open(path,'rb') as f:
        h=f.read(176)
        if h[:4]!=b'$FL2':
            raise ValueError('Expected SPSS $FL2')
        size,compression,n=struct.unpack_from('<iii',h,68)
        bias=struct.unpack_from('<d',h,84)[0]
        variables=[];extras=[]
        while True:
            raw=f.read(4)
            if len(raw)!=4: raise ValueError('Metadata ends unexpectedly')
            typ=struct.unpack('<i',raw)[0]
            if typ==2:
                buf=f.read(28)
                vtype,haslabel,nmiss,printfmt,writefmt=struct.unpack_from('<iiiii',buf,0)
                name=buf[20:28].rstrip(b' \x00').decode('latin-1')
                if haslabel:
                    length=struct.unpack('<i',f.read(4))[0]
                    f.seek(length+(-length)%4,1)
                if nmiss: f.seek(abs(nmiss)*8,1)
                variables.append((name,vtype))
            elif typ==3:
                count=struct.unpack('<i',f.read(4))[0]
                for i in range(count):
                    f.seek(8,1)
                    length=f.read(1)[0]
                    f.seek(length+(-length-1)%8,1)
            elif typ==4:
                count=struct.unpack('<i',f.read(4))[0]
                f.seek(count*4,1)
            elif typ==6:
                count=struct.unpack('<i',f.read(4))[0]
                f.seek(count*80,1)
            elif typ==7:
                sub,sz,count=struct.unpack('<iii',f.read(12))
                payload=f.read(sz*count)
                extras.append((sub,sz,count,payload))
            elif typ==999:
                extra=f.read(4)
                return {'case_size':size,'compression':compression,'cases':n,'bias':bias,'variables':variables,'extras':extras,'offset':f.tell()}
            else:raise ValueError(f'Unknown record type {typ} offset {f.tell()-4}')

if __name__=='__main__':
    for name in ['CAP_100E_URBANO_RURAL_1.sav','CAP_200_URBANO_RURAL_2.sav','CAP_300_URBANO_RURAL_3.sav']:
        r=inspect_sav('/mnt/data/'+name)
        print(name,'size',r['case_size'],'n',r['cases'],'offset',r['offset'],'vars',len(r['variables']),'extras',[(x[0],x[1],x[2]) for x in r['extras']]);print('first 45',r['variables'][:45]);print('subtype 13',[(x[0],x[3][:250]) for x in r['extras'] if x[0] in [13,14,16,21]])


def read_sav(path, usecols=None, maxrows=None):
    import numpy as np
    import pandas as pd
    meta=inspect_sav(path)
    longnames={}
    for sub,_,_,payload in meta['extras']:
        if sub==13:
            for item in payload.decode('utf-8',errors='replace').strip('\x00').split('\t'):
                if '=' in item:
                    a,b=item.split('=',1)
                    longnames[a]=b
    sources=[]
    chosen=set(usecols or [])
    for i,(short,kind) in enumerate(meta['variables']):
        if kind==-1:continue
        name=longnames.get(short,short).upper()
        if usecols is None or name in chosen:
            length=1 if kind==0 else (kind+7)//8
            sources.append((name,i,kind,length))
    if usecols is not None:
        missing=chosen-set(s[0] for s in sources)
        if missing:raise KeyError(f'{Path(path).name}: missing {sorted(missing)}')
    slots=[None]*meta['case_size']
    for name,idx,kind,length in sources:
        for j in range(length):slots[idx+j]=(name,kind,j)
    vals={x[0]:[] for x in sources}
    n_expected=None
    for sub,_,_,payload in meta['extras']:
        if sub==16:
            chunks=[int.from_bytes(payload[i:i+8],byteorder='little') for i in range(0,len(payload),8)]
            if len(chunks)>1:n_expected=chunks[1]
    if maxrows is not None:n_expected=min(n_expected or maxrows,maxrows)
    with open(path,'rb',buffering=16*1024*1024) as f:
        f.seek(meta['offset'])
        if meta['compression']!=1:raise NotImplementedError('Expected compressed $FL2')
        ctrl=b'';cc=8; read=f.read
        row_idx=0
        while n_expected is None or row_idx<n_expected:
            row={}; accum={}
            i=0
            while i<meta['case_size']:
                if cc==8:
                    ctrl=read(8)
                    if len(ctrl)!=8:break
                    cc=0
                code=ctrl[cc];cc+=1
                if code==0:continue
                if code==252:break
                info=slots[i]
                if code==253:
                    raw=read(8)
                    if len(raw)!=8:break
                    if info is not None:
                        nm,kind,j=info
                        if kind==0:
                            z=struct.unpack('<d',raw)[0]
                            row[nm]=np.nan if (z<=-1e307 or not np.isfinite(z)) else z
                        else:accum.setdefault(nm,[]).append(raw)
                elif code==254:
                    if info is not None:
                        nm,kind,j=info
                        if kind==0:row[nm]=np.nan
                        else:accum.setdefault(nm,[]).append(b' '*8)
                elif code==255:
                    if info is not None:
                        nm,kind,j=info
                        if kind==0:row[nm]=np.nan
                        else:accum.setdefault(nm,[]).append(b' '*8)
                else:
                    if info is not None:
                        nm,kind,j=info
                        if kind==0:row[nm]=float(code-meta['bias'])
                        else:accum.setdefault(nm,[]).append(b' '*8)
                i+=1
            if i<meta['case_size']:
                if n_expected is not None:raise ValueError(f'Truncated case {row_idx}, slot {i}')
                break
            for name,_,kind,_ in sources:
                if kind!=0:
                    row[name]=b''.join(accum.get(name,[])).decode('cp1252',errors='replace').rstrip(' \x00')
                vals[name].append(row.get(name,np.nan))
            row_idx+=1
    return pd.DataFrame(vals),meta
