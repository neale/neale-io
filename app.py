from flask import Flask, request, jsonify, render_template, url_for
import requests
import csv
from cppn import CPPN, run_cppn


APPNAME = 'NealeFlaskCPPNSite'
app = Flask(__name__, static_url_path='', static_folder='static')
app.config.update(APPNAME=APPNAME)



@app.route('/')
def home():
 #   return "sup, this is the main <h1>Hi</h1>"
 return render_template("index.html")


@app.route('/test')
def test():
    return render_template("new.html")


@app.route('/cppn')
def cppn_viewer():
    img = 'images/12.png'
    return render_template('cppn.html',
                            data=img)


@app.route("/<name>")
def user(name):
    return f"Hello {name}"


#@app.route('/sample', methods=['POST'])
#def sample()
    

if __name__ == '__main__':
    app.run(debug=True)
