"""Provider-specific credentials and request contracts; never cross-fallback keys."""
from dataclasses import dataclass
import hashlib
import json
import os
from urllib.parse import urlparse

PROVIDERS=('minimax','openai','qwen','custom')

@dataclass(frozen=True)
class ProviderSettings:
    provider:str
    model:str
    base_url:str
    key_env:str
    api_style:str
    token_parameter:str
    extra_body:dict

    @property
    def api_key(self):return os.getenv(self.key_env,'')

    @property
    def namespace(self):
        if self.provider=='minimax' and self.model in ('MiniMax-M2.7','MiniMax-M2.7-highspeed') and self.base_url=='https://api.minimax.cn/v1':
            return 'minimax-m27-legacy-v1'
        h=hashlib.sha256((self.base_url+'\n'+self.model+'\n'+self.api_style).encode()).hexdigest()[:12]
        return self.provider+'-'+h

    def public(self):
        return {'provider':self.provider,'model':self.model,'base_url':self.base_url,
                'api_style':self.api_style,'credential_configured':bool(self.api_key and self.api_key!='replace_with_your_key'),
                'key_env':self.key_env,'cache_namespace':self.namespace}

def get_settings(provider=None,audit=False):
    name=provider or (os.getenv('LLM_AUDIT_PROVIDER') if audit else None) or os.getenv('LLM_PROVIDER','minimax')
    if name not in PROVIDERS:raise ValueError('LLM_PROVIDER must be minimax, openai, qwen or custom')
    spec={
        'minimax':('MINIMAX','https://api.minimax.cn/v1','MiniMax-M2.7-highspeed','MINIMAX_API_KEY'),
        'openai':('OPENAI','https://api.openai.com/v1','','OPENAI_API_KEY'),
        'qwen':('QWEN','https://dashscope.aliyuncs.com/compatible-mode/v1','qwen-plus','DASHSCOPE_API_KEY'),
        'custom':('CUSTOM','','','CUSTOM_API_KEY'),
    }
    prefix,base,model,key_env=spec[name]
    base=os.getenv(prefix+'_BASE_URL',base).rstrip('/')
    model=os.getenv(prefix+'_MODEL',model)
    style=os.getenv(prefix+'_API_STYLE','chat')
    if style not in ('chat','responses'):raise ValueError('API_STYLE must be chat or responses')
    param=os.getenv(prefix+'_TOKEN_PARAMETER','max_completion_tokens' if name=='openai' else 'max_tokens')
    if param not in ('max_tokens','max_completion_tokens'):raise ValueError('Unsupported token limit parameter')
    url=urlparse(base)
    if base and (url.scheme not in ('http','https') or not url.netloc or url.username or url.password or url.query or url.fragment):
        raise ValueError('Base URL must be an HTTP(S) endpoint without credentials, query or fragment')
    try:extra=json.loads(os.getenv(prefix+'_EXTRA_BODY','{}'))
    except json.JSONDecodeError:raise ValueError('EXTRA_BODY must be a JSON object') from None
    if not isinstance(extra,dict) or set(extra)&{'model','messages','input','instructions','api_key','key','stream','max_tokens','max_completion_tokens','max_output_tokens'}:
        raise ValueError('EXTRA_BODY cannot override identity, input, authentication or output limits')
    return ProviderSettings(name,model,base,key_env,style,param,extra)

def request_payload(cfg,system,user,max_tokens):
    if not cfg.model or not cfg.base_url:raise ValueError('Configure the selected provider model and base URL')
    messages=[{'role':'system','content':system},{'role':'user','content':user}]
    if cfg.api_style=='responses':
        return '/responses',{'model':cfg.model,'input':messages,'max_output_tokens':max_tokens,**cfg.extra_body}
    return '/chat/completions',{'model':cfg.model,'messages':messages,cfg.token_parameter:max_tokens,**cfg.extra_body}

def response_text(cfg,data):
    if cfg.api_style=='responses':
        return ''.join(part.get('text','') for item in data.get('output',[]) if item.get('type')=='message'
                       for part in item.get('content',[]) if part.get('type')=='output_text')
    choices=data.get('choices',[])
    if not choices:return ''
    content=choices[0].get('message',{}).get('content','')
    if isinstance(content,list):return ''.join(p.get('text','') for p in content if p.get('type')=='text')
    return content or ''
