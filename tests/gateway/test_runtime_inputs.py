"""No providers: real receipt store, SessionDB and Runtime HTTP control."""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from hermes_state import SessionDB
from gateway.runtime_inputs import RuntimeInputs
from gateway.runtime_contract_models import RuntimeUserInput
from gateway.runtime_execution import ExecutionJournal, Execution
from gateway.api_server_runtime import APIServerRuntimeMixin


def setup_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "input-state.db")
    db.create_session(session_id="input-test", source="test", model="test-model")
    events = []
    session = SimpleNamespace(
        run_id="run_input", session_db=db, lock=threading.RLock(),
        finished=threading.Event(), interrupted=threading.Event(),
        emit=lambda kind, payload: events.append((kind,payload)), agent_ref=[None],
    )
    class Agent:
        def _persist_session(self, messages, history):
            for message in messages:
                identity = message.get("message_id")
                if identity and not db.has_platform_message_id("input-test", identity):
                    db.append_message(session_id="input-test", role=message["role"], content=message["content"], platform_message_id=identity)
    session.agent_ref[0] = Agent()
    session.inputs = RuntimeInputs(session)
    return session, db, events


def test_input_replay_is_once_and_does_not_rewrite_prior_context(tmp_path):
    session, db, events = setup_session(tmp_path)
    try:
        body = RuntimeUserInput(message_id="m1", text="change direction", attachment_asset_ids={"user_upload":["asset_test"]})
        session.inputs.accept(body)
        session.inputs.accept(body)
        messages = [{"role":"user","content":"original"},{"role":"assistant","content":"working"}]
        assert session.inputs.consume(messages, [])
        assert messages[:2] == [{"role":"user","content":"original"},{"role":"assistant","content":"working"}]
        assert "asset_test" in messages[-1]["content"]
        assert [kind for kind,_ in events] == ["input_received","input_received","input_applied"]
        session.inputs.accept(body)
        assert not session.inputs.consume(messages, [])
        assert len(db.get_messages("input-test")) == 1
        with pytest.raises(ValueError, match="different content"):
            session.inputs.accept(RuntimeUserInput(message_id="m1",text="another"))
    finally:
        db.close()


def test_pending_input_survives_suspension_and_applied_input_is_not_replayed(tmp_path):
    session, db, events = setup_session(tmp_path)
    try:
        session.inputs.accept(RuntimeUserInput(message_id="m1",text="next boundary"))
        resumed = RuntimeInputs(session)
        messages=[]
        assert resumed.consume(messages, [])
        assert not RuntimeInputs(session).consume(messages, [])
        assert len(messages)==1
    finally: db.close()


def test_finish_and_new_input_are_atomic(tmp_path):
    session, db, events = setup_session(tmp_path)
    try:
        messages=[]
        barrier=threading.Barrier(2)
        def accept():
            barrier.wait()
            try:
                session.inputs.accept(RuntimeUserInput(message_id="racing",text="late change"))
                return True
            except ValueError:
                return False
        def finish():
            barrier.wait()
            return session.inputs.consume(messages, [], finishing=True)
        with ThreadPoolExecutor(max_workers=2) as executor:
            accepting, finishing=executor.submit(accept), executor.submit(finish)
            accepted, consumed=accepting.result(), finishing.result()
        assert accepted == consumed
        if accepted:
            assert len(messages)==1
            assert not session.inputs.consume(messages, [], finishing=True)
        with pytest.raises(ValueError, match="no longer accepting"):
            session.inputs.accept(RuntimeUserInput(message_id="later",text="cannot disappear"))
    finally: db.close()


@pytest.mark.asyncio
async def test_input_http_fences_execution_and_lease(tmp_path):
    session, db, events=setup_session(tmp_path)
    journal=ExecutionJournal(db)
    journal.create("exec_input","run_input","digest")
    journal.claim_subscription("exec_input",2)
    execution=Execution("exec_input","run_input",journal,session=session)
    class Adapter(APIServerRuntimeMixin):
        async def _authenticate_runtime_request(self,request): return None
    adapter=Adapter()
    adapter._execution_registry()["exec_input"]=execution
    app=web.Application()
    app.router.add_post("/v1/runtime/runs/{run_id}/inputs",adapter._handle_runtime_input)
    try:
        async with TestClient(TestServer(app)) as client:
            body={"message_id":"http","text":"hello"}
            for run,epoch in [("other",2),("run_input",1)]:
                response=await client.post(f"/v1/runtime/runs/{run}/inputs?execution_id=exec_input&lease_epoch={epoch}",json=body)
                assert response.status==409
            target="/v1/runtime/runs/run_input/inputs?execution_id=exec_input&lease_epoch=2"
            for _ in range(2): assert (await client.post(target,json=body)).status==204
            assert len(events)==2
            session.inputs.consume([],[],finishing=True)
            session.inputs.consume([],[],finishing=True)
            assert (await client.post(target,json={"message_id":"late","text":"late"})).status==409
            assert (await client.post(target,json={**body,"principal":"forged"})).status==409
    finally: db.close()


def test_real_agent_consumes_input_arriving_during_final_response(tmp_path, monkeypatch):
    """Actual Agent loop and persistence, with a loopback fake model only."""
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from run_agent import AIAgent
    session, db, events = setup_session(tmp_path)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            is_task = any('Original task' in str(m.get('content','')) for m in request.get('messages', []))
            if is_task:
                requests.append(request)
            if is_task and len(requests) == 1:
                session.inputs.accept(RuntimeUserInput(message_id='during_final', text='Use the revised direction'))
            content = 'initial answer' if len(requests) == 1 else 'revised answer'
            self.send_response(200)
            if request.get('stream'):
                self.send_header('Content-Type','text/event-stream'); self.end_headers()
                for delta, finish in [({'role':'assistant','content':content}, None), ({},'stop')]:
                    self.wfile.write(('data: '+json.dumps({'id':'test','choices':[{'index':0,'delta':delta,'finish_reason':finish}]})+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
            else:
                self.send_header('Content-Type','application/json'); self.end_headers()
                self.wfile.write(json.dumps({'id':'test','choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}).encode())
    server=HTTPServer(('127.0.0.1',0), Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        agent=AIAgent(api_key='test-key', base_url=f'http://127.0.0.1:{server.server_port}/v1',
            provider='openai-compat', model='test-model', max_iterations=4, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            platform='cli', session_db=db, session_id='input-test')
        session.agent_ref[0]=agent
        agent._runtime_input_boundary=session.inputs.consume
        agent._require_incremental_session_persistence=True
        result=agent.run_conversation('Original task')
        assert len(requests)==2
        assert any(m.get('role')=='user' and m.get('content')=='Use the revised direction' for m in requests[1]['messages'])
        assert any(m.get('content')=='revised answer' for m in db.get_messages('input-test'))
        assert [kind for kind,_ in events]==['input_received','input_applied']
        assert not session.inputs.accepting
    finally:
        server.shutdown(); thread.join(timeout=2); server.server_close(); db.close()


def test_context_activity_keeps_native_return_and_failure():
    from gateway.runtime_progress import bind_context_activity
    events=[]
    compressor=SimpleNamespace(compression_count=0)
    def compress(*args, **kwargs):
        compressor.compression_count+=1
        return (['native history'], 'native system')
    agent=SimpleNamespace(_compress_context=compress, context_compressor=compressor)
    session=SimpleNamespace(emit=lambda kind,payload:events.append((kind,payload)))
    bind_context_activity(agent,session)
    assert agent._runtime_compression_observer(compress)==(['native history'],'native system')
    assert events[-1][1]['output']=={'compressed':True,'compression_count':1}
    def failure(*args,**kwargs): raise RuntimeError('test failure')
    agent._compress_context=failure
    bind_context_activity(agent,session)
    with pytest.raises(RuntimeError,match='test failure'): agent._runtime_compression_observer(failure)
    assert events[-1][1]['status']=='failed'
