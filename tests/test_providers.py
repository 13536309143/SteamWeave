import asyncio
import json
import pytest
from sg_agents.llm_settings import get_settings,request_payload,response_text
from sg_agents import provider

@pytest.mark.parametrize('name,key_env,param',[('minimax','MINIMAX_API_KEY','max_tokens'),('openai','OPENAI_API_KEY','max_completion_tokens'),('qwen','DASHSCOPE_API_KEY','max_tokens'),('custom','CUSTOM_API_KEY','max_tokens')])
def test_request_contracts_and_credential_isolation(monkeypatch,name,key_env,param):
    monkeypatch.setenv('OPENAI_MODEL','user-gpt-model');monkeypatch.setenv('CUSTOM_MODEL','local-model');monkeypatch.setenv('CUSTOM_BASE_URL','http://localhost:9999/v1')
    cfg=get_settings(name);suffix,payload=request_payload(cfg,'system','user',2048)
    assert cfg.key_env==key_env and suffix=='/chat/completions'
    assert payload[param]==2048 and 'api_key' not in payload
    assert response_text(cfg,{'choices':[{'message':{'content':'ok'}}]})=='ok'

def test_responses_protocol(monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL','user-gpt-model');monkeypatch.setenv('OPENAI_API_STYLE','responses')
    cfg=get_settings('openai');suffix,payload=request_payload(cfg,'system','user',1024)
    assert suffix=='/responses' and payload['max_output_tokens']==1024
    assert response_text(cfg,{'output':[{'type':'reasoning','content':[]},{'type':'message','content':[{'type':'output_text','text':'ok'}]}]})=='ok'

def test_cache_namespaces_do_not_mix(monkeypatch):
    monkeypatch.setenv('OPENAI_MODEL','A');a=get_settings('openai').namespace
    monkeypatch.setenv('OPENAI_MODEL','B');b=get_settings('openai').namespace
    assert a!=b and a!=get_settings('minimax').namespace

def test_foreign_provider_never_uses_minimax_key(monkeypatch):
    monkeypatch.setenv('MINIMAX_API_KEY','present-but-not-for-openai')
    monkeypatch.setenv('OPENAI_MODEL','user-gpt-model');monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(provider.ProviderError,match='OPENAI_API_KEY'):
        asyncio.run(provider.call_model('s','u',provider='openai'))

@pytest.mark.parametrize('name',['minimax','openai','qwen','custom'])
def test_transport_with_mock_server(monkeypatch,name):
    monkeypatch.setenv('OPENAI_MODEL','user-gpt-model');monkeypatch.setenv('CUSTOM_MODEL','local-model');monkeypatch.setenv('CUSTOM_BASE_URL','http://localhost:9999/v1')
    cfg=get_settings(name);monkeypatch.setenv(cfg.key_env,'test-key')
    seen={}
    class Response:
        status_code=200
        def json(self):return {'choices':[{'message':{'content':'ok'}}],'usage':{'total_tokens':1}}
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,headers,json):seen.update(url=url,payload=json,auth=headers['Authorization']);return Response()
    monkeypatch.setattr(provider.httpx,'AsyncClient',Client)
    content,usage=asyncio.run(provider.call_model('s','u',provider=name))
    assert content=='ok' and seen['auth']=='Bearer test-key'
    assert seen['url'].startswith(cfg.base_url)

def test_error_response_never_exposes_response_or_key(monkeypatch):
    monkeypatch.setenv('MINIMAX_API_KEY','test-secret')
    class Response:status_code=401
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,*args,**kwargs):return Response()
    monkeypatch.setattr(provider.httpx,'AsyncClient',Client)
    with pytest.raises(provider.ProviderError) as e:asyncio.run(provider.call_model('s','u'))
    assert '401' in str(e.value) and 'test-secret' not in str(e.value)
