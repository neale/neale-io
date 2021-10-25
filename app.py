import os
import re
import csv
import json, jsonify
import requests
import numpy as np
from glob import glob
from time import sleep
from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask import session

from cppn import CPPN, run_cppn


APPNAME = 'NealeFlaskCPPNSite'
app = Flask(__name__, static_url_path='', static_folder='static')
app.config.update(APPNAME=APPNAME)
app.secret_key = str(np.random.randint(int(1e9)))
app.debug=True


def init_cppn(uid):
    return CPPN(**session['cppn_config'])
    

@app.route('/home')
def home():
    print ('home')
    try:
        clear_images(session['uid'], 'png')
        clear_images(session['uid'], 'tiff')
    except Exception as e:
        print (e)

    return render_template("index.html")


@app.route('/cppn')
def cppn_viewer():
    # start CPPN 
    if 'uid' not in session.keys():
        uid = np.random.randint(int(1e9))
        session['uid'] = uid
    if 'cppn_config' not in session.keys():
        print('creating cppn config')
        session['cppn_config'] = {
            'z_dim'    : 8,
            'n_samples': 1,
            'x_dim': 512,
            'y_dim': 512,
            'c_dim': 3,
            'z_scale': 10,
            'layer_width': 32,
            'interpolation': 10,
            'exp_name': 'static/assets/cppn_images',
            'seed': 1234567890,
            'uid': session['uid'],
        }
    
    session['landing_img'] = 'images/12.png'
    
    print ('cppn z val ', session['cppn_config']['z_dim'])
    slider_vals = {
        'z_slider_init'            : session['cppn_config']['z_dim'],
        'layer_width_slider_init'  : session['cppn_config']['layer_width'],
        'z_scale_slider_init'      : session['cppn_config']['z_scale'],
        'interpolation_range_init' : session['cppn_config']['interpolation'],}
    
    return render_template('cppn.html',
                            data=session['landing_img'],
                            slider_vals=slider_vals)


def sort_fn_nums(fns):
    """Sorts files and returns the largest, smallest values according
        to the number (index) at the end of the filename

    Args:
        fns (list): file names to be sorted

    Returns:
        f_min (str): smallest filename according to index.
        f_max (str): largest filename according to index.
    """
    fns_idx = lambda f: int(f.split('_')[-1].split('.')[0])
    sorted_fns = sorted(fns, key=fns_idx)
    print (sorted_fns)
    f_min = sorted_fns[0]
    f_max = sorted_fns[-1]
    return f_min, f_max


def clear_images(uid, suffix='png'):
    assert suffix in ['png', 'tiff']
    sample_dir = 'static/assets/cppn_images/{}'.format(uid)
    if suffix == 'png':
        img_in_dir = glob('{}/*.png'.format(sample_dir))
    elif suffix == 'tiff':
        img_in_dir = glob('{}/*.tiff'.format(sample_dir))
    print (f'clearing {suffix} files')
    for image in img_in_dir:
        os.remove(image)


def run_image(cppn, uid, autosave):
    sample, fn_suff = run_cppn(cppn, uid, autosave=autosave)
    return sample, fn_suff


@app.route('/generate-image-random', methods=['GET', 'POST'])
def generate_image_random():
    pass

@app.route('/generate-image', methods=['GET', 'POST'])
def generate_image():
    """
    Generates and renders a single image (no ajax jet so page is refreshed)
    """
    autosave = True
    if request.method == 'GET':
        return redirect("/cppn")

    elif request.method == 'POST':
        if 'cppn_config' not in session.keys():
            return redirect("/cppn")
        
        print ('gen image', session['cppn_config'])
        cppn = init_cppn(uid=session['uid'])
        if cppn.generator is None:
            cppn.init_generator()
        uid = session['uid']
        sample_dir = 'static/assets/cppn_images/{}'.format(uid)
        print ('running with Z', cppn.z_dim)
        if len(glob('{}/*.png'.format(sample_dir))) > 100:
            clear_images(uid, 'png') 
            clear_images(uid, 'tiff') 
        sample, fn_suff = run_image(cppn, uid, autosave)
        assert sample is not None, "sample is None"
        if not autosave: # image did not get saved under unique ID
            path = f'{cppn.exp_name}/{cppn.uid}/sample'
            print (f'saving sample to path: {path}')
            cppn._write_image(path=path, x=sample, suffix='PNG')
        images_in_dir = glob('{}/*.png'.format(sample_dir))
        if len(images_in_dir) > 1:
            _, latest_image = sort_fn_nums(images_in_dir)
        else:
            latest_image = images_in_dir[0]
        print (f'new image from _generate_image: {latest_image}')
        latest_image = latest_image.split('static/')[1]
        return jsonify({'img': latest_image})

    else:
        return jsonify({'img': 'image_12.png'})


@app.route('/slider-control', methods=['GET', 'POST'])
def slider_control():
    print ('request.form:', request.form)
    if request.method == 'GET':
        print ('get request')
        return jsonify({'nothing': 0})

    elif request.method == 'POST':
        #cppn, uid = CPPN_cache[0]
        uid  = session['uid']
        if request.form.get('z_range'):
            new_z = int(request.form.get('z_range'))
            print ('new ', new_z, 'old_z', session['cppn_config']['z_dim'])
            if new_z != session['cppn_config']['z_dim']:
                session['cppn_config']['z_dim'] = new_z
                session.modified = True
            print ('changed to: ', session['cppn_config']['z_dim'])
            return jsonify({'z': new_z})

        if request.form.get('net_width_range'):
            new_width = int(request.form.get('net_width_range'))
            if new_width != session['cppn_config']['layer_width']:
                session['cppn_config']['layer_width'] = new_width
                session.modified = True
            return jsonify({'net_width': new_width})

        if request.form.get('z_scale_range'):
            new_z_scale = float(request.form.get('z_scale_range'))
            if new_z_scale != session['cppn_config']['z_scale']:
                session['cppn_config']['z_scale'] = new_z_scale
                session.modified = True
            return jsonify({'z_scale': new_z_scale})

        if request.form.get('interpolation_range'):
            new_interpolate = int(request.form.get('interpolation_range'))
            if new_interpolate != session['cppn_config']['interpolation']:
                session['cppn_config']['interpolation'] = new_interpolate
                session.modified = True
            return jsonify({'interpolation': new_interpolate})
        else:
            return jsonify({'nothing': 0})


@app.route("/<name>")
def user(name):
    return redirect("/")

@app.route("/")
def root():
    return redirect("/home")


if __name__ == '__main__':
    app.run(debug=True)
