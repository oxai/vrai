from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        print(self.path)
        try:
            function, argument = self.path.split("/")[1:]
            if function == "obs":
                open("observation.txt", "w").write(argument)
                self._set_headers()
                self.wfile.write(bytes(str("hi"), "utf-8"))
            if function == "act":
                action_response = "".join(["{:.7s}".format('{:0.5f}'.format(x)) for x in 4*np.random.rand(20,6).flatten()-2])
                # print(action_response)
                self._set_headers()
                # self.wfile.write(bytes(str("hi"), "utf-8"))
                self.wfile.write(bytes(action_response, "utf-8"))
        except ValueError:
            self._set_headers()
            self.wfile.write(bytes(str("hi"), "utf-8"))

        # self.wfile.write(bytes("<html><body><h1>"+str(classification)+"</h1></body></html>", "utf-8"))

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
