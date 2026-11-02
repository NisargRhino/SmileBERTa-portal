"""Local dev server for frontend/ that mimics Render's static hosting:
/builder serves builder.html, / serves index.html.

    python3 serve_frontend.py   # http://localhost:8080
"""
import http.server
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def send_head(self):
        path = self.path.split('?', 1)[0].split('#', 1)[0]
        if path != '/' and '.' not in os.path.basename(path) and os.path.isfile(os.path.join(ROOT, path.lstrip('/') + '.html')):
            self.path = path + '.html' + self.path[len(path):]
        return super().send_head()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f'Serving frontend on http://localhost:{port}')
    http.server.ThreadingHTTPServer(('', port), Handler).serve_forever()
