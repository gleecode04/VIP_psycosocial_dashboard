from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import checkin_bridge
from live_dashboard_backend import LiveDashboardBridge
from posture_live_bridge import bridge as posture_bridge

ROOT = Path(__file__).resolve().parent
PORT = 8000
LIVE_STREAM_URL = 'http://127.0.0.1:7001/latest'  # matches avro_stream_poller_with_ml.py / mock_poller.py
POSTURE_DEBUG_LOG_PATH = ROOT / 'posture_connection_debug.log'

bridge = LiveDashboardBridge(ROOT / 'data' / 'dashboard_data.json', live_url=LIVE_STREAM_URL, poll_seconds=5)
bridge.start()

_posture_debug_log_lock = threading.Lock()


def _append_posture_debug_log(lines):
    with _posture_debug_log_lock, open(POSTURE_DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{datetime.now().isoformat(timespec="milliseconds")} {line}\n')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/dashboard-data':
            return self._send_json(bridge.get_dashboard_data())

        if path == '/api/live-status':
            data = bridge.get_dashboard_data()
            return self._send_json({
                'liveStatus': data.get('liveStatus', {}),
                'liveLatestSample': data.get('liveLatestSample'),
            })

        if path == '/api/posture/latest':
            return self._send_json(posture_bridge.get_latest())

        if path == '/api/checkin/pending':
            query = parse_qs(parsed.query)
            participant_id = (query.get('participant_id') or [None])[0]
            if not participant_id:
                context = bridge.get_live_context()
                participant_id = context['participant_id'] if context else 'LIVE'
            return self._send_json(checkin_bridge.bridge.get_pending(participant_id))

        if path == '/api/checkin/stats':
            return self._send_json(checkin_bridge.bridge.get_stats())

        if path == '/':
            self.path = '/psychosocial_dashboard.html'
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/posture/connection':
            body = self._read_json_body()
            posture_bridge.set_connected(bool(body.get('connected')))
            return self._send_json({'ok': True})

        if path == '/api/posture/reading':
            body = self._read_json_body()
            try:
                posture_bridge.add_reading(
                    float(body['t']), float(body['ax']), float(body['ay']), float(body['az'])
                )
            except (KeyError, TypeError, ValueError):
                return self._send_json({'ok': False, 'error': 'expected {t, ax, ay, az}'}, status=400)
            return self._send_json({'ok': True})

        if path in ('/api/posture/session/start', '/api/posture/session/restart'):
            posture_bridge.start_session()
            return self._send_json({'ok': True})

        if path == '/api/posture/debug-log':
            body = self._read_json_body()
            lines = body.get('lines') or ([body['line']] if 'line' in body else [])
            _append_posture_debug_log(str(line) for line in lines)
            return self._send_json({'ok': True})

        if path == '/api/checkin/answer':
            body = self._read_json_body()
            question_id = body.get('question_id')
            if not question_id:
                return self._send_json({'ok': False, 'error': 'question_id is required'}, status=400)
            record = checkin_bridge.bridge.answer(
                question_id, body.get('stressed'), body.get('mental'), body.get('free_text'),
            )
            if record is None:
                return self._send_json(
                    {'ok': False, 'error': 'no open question with that id (it may have already expired)'},
                    status=404,
                )
            return self._send_json({'ok': True, 'record': record})

        if path == '/api/checkin/self-report':
            body = self._read_json_body()
            context = bridge.get_live_context()
            if context is None:
                return self._send_json(
                    {'ok': False, 'error': 'no live sample yet -- start the wristband stream first'}, status=409,
                )
            record = checkin_bridge.bridge.self_report(
                context['participant_id'], body.get('stressed'), body.get('mental'), body.get('free_text'),
                context['stress_label'], context['mental_label'], context['features_snapshot'],
            )
            return self._send_json({'ok': True, 'record': record})

        return self._send_json({'ok': False, 'error': 'not found'}, status=404)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        super().end_headers()

    def _read_json_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode('utf-8'))

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    url = f'http://127.0.0.1:{PORT}/psychosocial_dashboard.html'
    print(f'Serving integrated dashboard at {url}')
    print(f'Expecting live stream at {LIVE_STREAM_URL}')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
