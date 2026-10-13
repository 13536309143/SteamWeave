"""Architecture reconstruction, not an exact reproduction of undisclosed training code."""
import math
import time
import numpy as np
import torch
from torch import nn
from PIL import Image
from .glove import tokenize

class MultiBranchDNN(nn.Module):
    def __init__(self,embedding_matrix):
        super().__init__()
        self.image_cnn=nn.Sequential(nn.Conv2d(1,4,4,padding='same'),nn.ReLU(),nn.MaxPool2d(4,ceil_mode=True),
            nn.Conv2d(4,4,4,padding='same'),nn.ReLU(),nn.MaxPool2d(4,ceil_mode=True),
            nn.Conv2d(4,8,4,padding='same'),nn.ReLU(),nn.MaxPool2d(4,ceil_mode=True),nn.Flatten(),
            nn.Linear(8*6*4,4),nn.Sigmoid())
        self.embedding=nn.Embedding.from_pretrained(torch.tensor(embedding_matrix),freeze=False,padding_idx=0)
        self.text_convs=nn.ModuleList([nn.Conv1d(100,32,k) for k in (2,3,4)])
        self.text_fc=nn.Sequential(nn.Linear(96,4),nn.Sigmoid())
        self.numeric=nn.Sequential(nn.Linear(4,4),nn.Sigmoid())
        self.output=nn.Linear(12,1)

    def forward(self,image,tokens,numeric):
        a=self.image_cnn(image)
        embedded=self.embedding(tokens).transpose(1,2)
        text=torch.cat([torch.relu(c(embedded)).amax(dim=-1) for c in self.text_convs],dim=1)
        b=self.text_fc(text);c=self.numeric(numeric)
        return self.output(torch.cat([a,b,c],dim=1)).squeeze(-1)

def prepare_inputs(records,features,glove_path):
    # uint8 uses ~480 MB for the full dataset rather than ~1.9 GB float32.
    images=np.zeros((len(records),1,325,194),dtype=np.uint8)
    for i,r in enumerate(records):
        if features['quality'][i,0]:
            with Image.open(r.image_path) as im:
                a=np.asarray(im.convert('L'))
            if a.shape!=(325,194):raise ValueError('Paper baseline expects 325x194 images')
            images[i,0]=a
    g=np.load(glove_path);words=g['words'].tolist();vocab={w:i+2 for i,w in enumerate(words)}
    tokens=np.zeros((len(records),27),dtype=np.int64)
    for i,r in enumerate(records):
        ws=tokenize(r.text)[:27];tokens[i,:len(ws)]=[vocab.get(w,1) for w in ws]
    return {'images':images,'tokens':tokens,'glove_vectors':g['vectors'],'glove_found':g['found'],
            'numeric':features['numeric'].astype(np.float32),'vocabulary':words}

def fit_predict(inputs,y,train_ix,val_ix,test_ix,seed,epochs=160,patience=20,batch_size=128,device='mps',progress=print):
    started=time.monotonic();torch.manual_seed(seed);np_rng=np.random.default_rng(seed)
    torch.set_num_threads(2)
    embedding=np_rng.normal(0,.05,size=(len(inputs['vocabulary'])+2,100)).astype(np.float32)
    embedding[0]=0;embedding[2:][inputs['glove_found']]=inputs['glove_vectors'][inputs['glove_found']]
    model=MultiBranchDNN(embedding).to(device)
    num_mean=inputs['numeric'][train_ix].mean(0);num_std=inputs['numeric'][train_ix].std(0);num_std[num_std<1e-6]=1
    normalized_num=((inputs['numeric']-num_mean)/num_std).astype(np.float32)
    ymean=float(y[train_ix].mean());ystd=max(float(y[train_ix].std()),1.)
    normalized_y=((y-ymean)/ystd).astype(np.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    def batch(ix):
        image=torch.from_numpy(inputs['images'][ix]).to(device=device,dtype=torch.float32)/255.
        tokens=torch.from_numpy(inputs['tokens'][ix]).to(device)
        num=torch.from_numpy(normalized_num[ix]).to(device)
        return image,tokens,num
    def prediction(ix):
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,len(ix),batch_size):
                parts.append(model(*batch(ix[start:start+batch_size])).cpu().numpy())
        return np.concatenate(parts)*ystd+ymean
    best=float('inf');best_state=None;best_epoch=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=np_rng.permutation(train_ix);loss_sum=torch.zeros((),device=device)
        for start in range(0,len(order),batch_size):
            ix=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            pred=model(*batch(ix));target=torch.from_numpy(normalized_y[ix]).to(device)
            loss=nn.functional.mse_loss(pred,target);loss.backward();optimizer.step();loss_sum+=loss.detach()*len(ix)
        val=prediction(val_ix);val_mse=float(np.mean((val-y[val_ix])**2))
        history.append({'epoch':epoch,'train_normalized_mse':float(loss_sum.cpu())/len(train_ix),'validation_mse_adu2':val_mse})
        if val_mse<best-1e-4:
            best=val_mse;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%10==0:progress(f'paper DNN epoch {epoch}: validation MAE={np.mean(np.abs(val-y[val_ix])):.3f}; best epoch={best_epoch}')
        if epoch-best_epoch>=patience:break
    model.load_state_dict(best_state);pred=prediction(test_ix)
    result={'prediction':pred,'history':history,'best_epoch':best_epoch,'best_validation_mse_adu2':best,
            'epochs_run':len(history),'training_seconds':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),
            'implementation':'paper-architecture-reconstruction','device':device,
            'assumptions':{'optimizer':'Adam','lr':.001,'batch_size':batch_size,'max_epochs':epochs,'patience':patience,
               'tokenization':'lowercase regex, truncate/pad to 27','embedding':'official GloVe 6B 100d, fine-tuned; missing vectors seed-initialized',
               'scaling':'training-only numeric z-score and target z-score; images /255; blank images zeroed',
               'head':'linear regression from concatenated 12-dimensional features'}}
    del model,optimizer
    if device=='mps':torch.mps.empty_cache()
    return result
