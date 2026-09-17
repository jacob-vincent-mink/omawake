"""Controlled protocol peer. Never loads native code or crashes intentionally."""
import json, socket, struct, sys
peer = socket.socket(socket.AF_UNIX)
peer.connect(sys.argv[-1])
mode = sys.argv[1]
def read_exact(n):
    data = b''
    while len(data) < n:
        part = peer.recv(n-len(data))
        if not part:
            raise EOFError()
        data += part
    return data
def receive():
    return json.loads(read_exact(struct.unpack('<I', read_exact(4))[0]))
def send(value):
    data = json.dumps(value).encode()
    peer.sendall(struct.pack('<I', len(data))+data)
receive()
if mode == 'startup-error':
    send({'Error':'controlled unavailable runtime'})
    sys.exit(0)
send({'Ready':{'contract':'test-encoder','execution_devices':'TEST'}})
try:
    while True:
        request = receive()
        if request == 'Start':
            send({'Utterances':[]})
        elif isinstance(request, dict) and 'Audio' in request:
            if mode == 'disconnect':
                break
            if mode == 'error':
                send({'Error':'controlled inference failure'})
            elif mode == 'oversized':
                peer.sendall(struct.pack('<I', 9000000))
                break
            elif mode == 'bad-embedding':
                send({'Utterances':[{'embedding':{'encoder_contract':'wrong','values':[], 'source_frames':1,'inference_ms':1.0,'execution_devices':'TEST'},'start_sample':0}]})
            else:
                send({'Utterances':[]})
        elif request == 'Finish':
            send({'Utterances':[]})
except (EOFError, BrokenPipeError):
    pass
