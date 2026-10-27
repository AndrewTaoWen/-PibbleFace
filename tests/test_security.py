import importlib.util
from pathlib import Path
import sys
import types


def test_local_camera_routes_reject_untrusted_hosts_and_cross_origin_reads(monkeypatch):
    # The security test never opens a camera or initializes a vision model.
    monkeypatch.setitem(sys.modules,'cv2',types.ModuleType('cv2'))
    monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    matcher=types.ModuleType('face_emoji_matcher')
    matcher.FaceEmojiMatcher=lambda:types.SimpleNamespace(emoji_database={})
    monkeypatch.setitem(sys.modules,'face_emoji_matcher',matcher)
    spec=importlib.util.spec_from_file_location('camera_test_app',Path(__file__).resolve().parents[1]/'app.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.app.test_client() as client:
        assert client.get('/get_emoji',base_url='http://attacker.example').status_code==400
        response=client.get('/get_emoji',headers={'Origin':'https://attacker.example'})
        assert response.status_code==200
        assert 'Access-Control-Allow-Origin' not in response.headers
        assert response.headers['Cross-Origin-Resource-Policy']=='same-origin'
        assert response.headers['X-Frame-Options']=='DENY'
        assert response.headers['Cache-Control']=='no-store'
