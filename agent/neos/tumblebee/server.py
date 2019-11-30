from http.server import BaseHTTPRequestHandler, HTTPServer

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        print(self.path)
        if self.path == "/take":
            if open("take.txt", "r").read().strip()=="yes":
                open("take.txt", "w").write("no")
            else:
                open("take.txt", "w").write("yes")
        if self.path == "/good":
            if open("reward.txt", "r").read().strip()!="0":
                open("reward.txt", "w").write("0")
            else:
                open("reward.txt", "w").write("1")
        if self.path == "/bad":
            if open("reward.txt", "r").read().strip()!="0":
                open("reward.txt", "w").write("0")
            else:
                open("reward.txt", "w").write("-1")
        self._set_headers()
        classification = open("classification.txt", "r").read()
        # self.wfile.write(bytes("<html><body><h1>"+str(classification)+"</h1></body></html>", "utf-8"))
        self.wfile.write(bytes(str(classification), "utf-8"))

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        self.wfile.write(bytes("<html><body><h1>POST!</h1></body></html>", "utf-8"))

def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

run(port=42069)
