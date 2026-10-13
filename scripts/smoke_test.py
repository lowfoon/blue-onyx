#!/usr/bin/env python3
"""Exercise a real CPU model, compatible API, readiness, and SIGTERM shutdown."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/debug/blue_onyx')
    parser.add_argument('--model', help='Optional existing ONNX model with YAML sidecar')
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve())
    with tempfile.TemporaryDirectory(prefix='blueonyx-smoke-') as folder:
        root = Path(folder)
        config = root / 'config.json'
        config.write_text('{}')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = dict(os.environ, BLUE_ONYX_MODEL_DIR=str(root / 'models'))
        command = [binary, '--config', str(config), '--port', str(port), '--force-cpu']
        if args.model:
            command += ['--model', str(Path(args.model).resolve())]
        log_path = root / 'service.log'
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                base = f'http://127.0.0.1:{port}'
                for _ in range(240):
                    if process.poll() is not None:
                        raise RuntimeError(f'Service exited during startup: {process.returncode}')
                    try:
                        with urllib.request.urlopen(base + '/health/ready', timeout=2) as reply:
                            if reply.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if 'Detector initialization failed' in log_path.read_text(errors='replace'):
                        raise RuntimeError('Detector initialization failed')
                    time.sleep(0.5)
                else:
                    raise RuntimeError('Detector readiness timed out')
                with urllib.request.urlopen(base + '/health/live', timeout=2) as reply:
                    assert reply.status == 200
                image = (Path(__file__).resolve().parents[1] / 'assets/dog_bike_car.jpg').read_bytes()
                body = (b'--smoke\r\nContent-Disposition: form-data; name="min_confidence"\r\n\r\n0.3\r\n'
                        b'--smoke\r\nContent-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n--smoke--\r\n')
                request = urllib.request.Request(base + '/v1/vision/detection', body,
                    {'Content-Type': 'multipart/form-data; boundary=smoke'})
                with urllib.request.urlopen(request, timeout=20) as reply:
                    result = json.load(reply)
                assert result['success'], result
                assert result['executionProvider'] == 'CPU', result
                assert result['count'] == len(result['predictions'])
                assert result['count'] > 0, result
                assert all(key in result for key in ['inferenceMs', 'analysisRoundTripMs', 'canUseGPU'])
                labels = sorted({prediction['label'] for prediction in result['predictions']})
                print(f'Real inference passed: {result["count"]} predictions, labels={labels}', flush=True)
                process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=20) == 0
                print('Readiness, compatible JSON, CPU inference, and SIGTERM shutdown passed.', flush=True)
            except Exception:
                print(log_path.read_text(errors='replace')[-10000:])
                raise
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == '__main__':
    main()
